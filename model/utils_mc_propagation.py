
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import logging
import logging.config
from scipy.stats import multivariate_normal as mvn

from model.utils import sparse_eigh_testing
from model.utils import covariance_matrix_projection

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
        
class MonteCarloResults:
    def __init__(self) -> None:
        self.predictions = None
        self.mean        = None
        self.cov         = None           
        
class MonteCarloPropagation:
    def __init__(self, ref_nb_sample, batch_size, use_subspace, subspace_threshold, subspace_max_dim, nb_class, is_mlp, *args, **kwargs) -> None:        
        self.ref_nb_sample      = ref_nb_sample
        self.batch_size         = batch_size
        self.use_subspace       = use_subspace
        self.subspace_threshold = subspace_threshold
        self.subspace_max_dim   = subspace_max_dim
        self.nb_class           = nb_class
        self.is_mlp             = is_mlp
        self.nb_batch           = self.compute_number_of_batches()
        self.last_batch_size    = self.compute_last_batch_size()         
    
    def compute_number_of_batches(self):
        return np.floor(self.ref_nb_sample / self.batch_size).astype(np.int32)
    
    def compute_last_batch_size(self):
        return self.ref_nb_sample % self.batch_size  
    
    @staticmethod
    def sample_from_mvn(mean, cov, nb_sample):     
        if len(tf.shape(cov)) == 2:
            cov = cov[tf.newaxis]

        _,_,cov = covariance_matrix_projection(cov[0], 1e-5)
        return mvn(mean[0], cov).rvs(int(nb_sample))

    @staticmethod
    def predict_mc_samples(model, mc_samples):
        return model.predict(mc_samples)
    
    def get_prediction_sample_mvn_subspace(self, model, subspace_mean, subspace_cov, mean, ev, nb_sample, orig_shape):
        zero_mean        = tf.zeros_like(subspace_mean)
        subspace_samples = self.sample_from_mvn(zero_mean, subspace_cov, nb_sample)
        mc_samples       = ev @ subspace_samples[:,:,tf.newaxis]
        mc_samples       = mc_samples[:,:,0].numpy() + mean
        if self.is_mlp:
            mc_samples       = tf.convert_to_tensor(mc_samples, dtype=tf.float32) 
        else:
            mc_samples       = tf.convert_to_tensor(np.reshape(mc_samples, (mc_samples.shape[0], orig_shape[0], orig_shape[1], orig_shape[2])), dtype=tf.float32) 
        
        mc_predictions = self.predict_mc_samples(model, mc_samples)
        return mc_predictions
    
    def get_prediction_sample_mvn(self, model, mean, cov, nb_sample, orig_shape):
        mc_samples      = self.sample_from_mvn(mean, cov, nb_sample)
        if self.is_mlp:
            mc_samples       = tf.convert_to_tensor(mc_samples, dtype=tf.float32) 
        else:
            mc_samples       = tf.convert_to_tensor(np.reshape(mc_samples, (mc_samples.shape[0], orig_shape[0], orig_shape[1], orig_shape[2])), dtype=tf.float32)         
        mc_predictions = self.predict_mc_samples(model, mc_samples)

        return mc_predictions    
    
    def _propagate_using_subspace(self, img, model, sample_mean, sample_cov, sample_weight):
        logger.info(f"MC samples generation using subspace (in progress)")
        mc_predictions  = np.zeros((self.ref_nb_sample, self.nb_class))
        ew, ev          = sparse_eigh_testing(sample_cov, sample_weight, self.subspace_max_dim)
        ew              = ew[np.newaxis,:]
        ev              = ev[np.newaxis,:,:]
        ev              = tf.cast(ev, tf.float32)
        mask            = ew > self.subspace_threshold
        ew              = tf.boolean_mask(ew, mask[0], axis=1)
        ev              = tf.boolean_mask(ev, mask[0], axis=2)                
        subspace_cov    = tf.linalg.diag(ew)
        subspace_mean   = tf.linalg.matrix_transpose(ev) @ sample_mean[:,:,tf.newaxis]
        subspace_mean   = subspace_mean[:,:,0]
        img_shape       = img.shape

        for i in range(self.nb_batch):
            cur_predictions = self.get_prediction_sample_mvn_subspace(
                model=model, 
                subspace_mean=subspace_mean, 
                subspace_cov=subspace_cov, 
                mean=sample_mean, 
                ev=ev, 
                nb_sample=self.batch_size, 
                orig_shape=img_shape
                )
            mc_predictions[self.batch_size*i:self.batch_size*(i+1)] = cur_predictions
        
        if self.last_batch_size > 0:
            cur_predictions = self.get_prediction_sample_mvn_subspace(
                model=model, 
                subspace_mean=subspace_mean, 
                subspace_cov=subspace_cov, 
                mean=sample_mean, 
                ev=ev, 
                nb_sample=self.last_batch_size, 
                orig_shape=img_shape
                )
            mc_predictions[self.batch_size*i:self.batch_size*(i+1)] = cur_predictions
        logger.info(f"MC samples generation using subspace (finished)")
        return mc_predictions
    
    def _propagate(self, img, model, sample_mean, sample_cov):
        logger.info(f"MC samples generation (in progress)")
        mc_predictions  = np.zeros((self.ref_nb_sample, self.nb_class))
        img_shape       = img.shape

        for i in range(self.nb_batch):
            cur_predictions = self.get_prediction_sample_mvn(model=model, mean=sample_mean, cov=sample_cov, nb_sample=self.batch_size, orig_shape=img_shape)
            mc_predictions[self.batch_size*i:self.batch_size*(i+1)] = cur_predictions

        if self.last_batch_size > 0:
            cur_predictions = self.get_prediction_sample_mvn(model=model, mean=sample_mean, cov=sample_cov, nb_sample=self.batch_size, orig_shape=img_shape)
            mc_predictions[self.batch_size*i:self.batch_size*(i+1)] = cur_predictions
        logger.info(f"MC samples generation (finished)")
        return mc_predictions
        
    def propagate(self, img, model, sample_mean, sample_cov, sample_weight):
        montecarloresults_obj = MonteCarloResults()
        
        if self.use_subspace:
            montecarloresults_obj.predictions = self._propagate_using_subspace(img, model, sample_mean, sample_cov, sample_weight)
        else:
            montecarloresults_obj.predictions = self._propagate(img, model, sample_mean, sample_cov)

        montecarloresults_obj.mean        = tf.reduce_mean(montecarloresults_obj.predictions, axis=0)
        montecarloresults_obj.cov         = tfp.stats.covariance(montecarloresults_obj.predictions)
        
        logger.info(f"shape of output mean {montecarloresults_obj.mean.shape}")
        logger.info(f"shape of output covariance matrix {montecarloresults_obj.cov.shape}")        
                       
        return montecarloresults_obj
