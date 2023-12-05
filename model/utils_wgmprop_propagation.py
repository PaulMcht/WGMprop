import numpy as np
import tensorflow as tf
import logging
import logging.config

from model.utils import eigh
from model.utils import sparse_eigh_testing, sparse_eigh
from model.enums import PropagationMethod

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class WGMpropResults:
    def __init__(self) -> None:
        self.gmm_out_weights = None
        self.gmm_out_means   = None
        self.gmm_out_vars    = None
        self.levels          = None
        self.final_criteria  = None
        self.ew              = None
        self.method          = None

class WGMpropPropagation:
    def __init__(self, batch_size, nb_burn, splitting_level, splitting_threshold, max_split, use_subspace, subspace_threshold, subspace_max_dim, use_tf_network_linearization, is_mlp, *args, **kwargs) -> None:
        self.use_subspace                 = use_subspace
        self.subspace_threshold           = subspace_threshold
        self.subspace_max_dim             = subspace_max_dim
        self.nb_burn                      = int(nb_burn)
        self.splitting_level              = splitting_level
        self.splitting_threshold          = splitting_threshold
        self.max_split                    = max_split
        self.use_tf_network_linearization = use_tf_network_linearization
        self.batch_size                   = batch_size
        self.is_mlp                       = is_mlp

        if self.splitting_level == 3:
            self._splitting_weights         = tf.convert_to_tensor(np.array([0.6364, 0.1818, 0.1818]), dtype=tf.float32)
            self._splitting_means           = tf.convert_to_tensor(np.array([0.0,    1.0579, -1.0579]), dtype=tf.float32)
            self._sigma_squared             = 0.7687**2 
        elif self.splitting_level == 5:
            self._splitting_weights         = tf.convert_to_tensor(np.array([0.4444, 0.2455,  0.2455, 0.0323,  0.0323]), dtype=tf.float32)
            self._splitting_means           = tf.convert_to_tensor(np.array([0.0,    0.9332, -0.9332, 1.9776, -1.9776]), dtype=tf.float32)
            self._sigma_squared             = 0.5654**2
        elif self.splitting_level == 7:
            self._splitting_weights         = tf.convert_to_tensor(np.array([0.3048, 0.2410,  0.2410, 0.0948,  0.0948, 0.0118,  0.0118]), dtype=tf.float32)
            self._splitting_means           = tf.convert_to_tensor(np.array([0.0,    0.7056, -0.7056, 1.4992, -1.4992, 2.4601, -2.4601]), dtype=tf.float32)        
            self._sigma_squared             = 0.4389**2

    def _compute_eigen_components_using_subspace(self, sample_cov, sample_weight):
        ew, ev                  = sparse_eigh(sample_cov, sample_weight, self.subspace_max_dim) #sparse_eigh_testing(sample_cov, sample_weight, self.subspace_max_dim)
        ew                      = ew[np.newaxis,:]
        ev                      = ev[np.newaxis,:,:]
        mask                    = ew > self.subspace_threshold
        ew                      = tf.boolean_mask(ew, mask[0], axis=1)
        ev                      = tf.boolean_mask(ev, mask[0], axis=2)

        return ew, ev

    @staticmethod
    def _compute_eigen_components(sample_cov, sample_weight):
        if len(tf.shape(sample_cov)) == 2:
            sample_cov = sample_cov[tf.newaxis]
            
        ew, ev                  = tf.map_fn(lambda x: eigh(x[0], x[1]), elems=(sample_cov, sample_weight))
        mask                    = ew > 1e-7
        ew                      = tf.boolean_mask(ew, mask[0], axis=1)
        ev                      = tf.boolean_mask(ev, mask[0], axis=2)
        return ew, ev        

    def _get_leveled_ews(self, ew, max_level=0):        
        
        # Store the original ew values in a TensorArray
        leveled_ews_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        leveled_ews_ta = leveled_ews_ta.write(0, ew)
        
        # Compute leveled_ews for each level and store the results in the TensorArray
        for i in range(1, max_level+1):
            highest_ew_idx  = tf.math.argmax(ew)
            updates         = tf.gather(ew, highest_ew_idx) * self._sigma_squared
            ew              = tf.tensor_scatter_nd_update(ew, [[highest_ew_idx]], [updates])
            leveled_ews_ta  = leveled_ews_ta.write(i, ew)
        
        # Return the TensorArray as a tensor with dynamic shape
        return leveled_ews_ta.stack()

    def _classic_level_split(self, weights, means, levels, leveled_ews, orig_ev, directions=None):
        """_summary_

        Args:
            weights (tensor): the weight
            means (tensor): the mean vector
            covs (tensor): the covariance matrix
            n (int): the number of split (3 / 5 or 7)
            ew (tensor): the eigenvalues of covs
            ev (tensor): the eigenvectors of covs

        Returns:
            list of tensor: the splitted weights, means, covs, eigenvalues and eigenvectors
        """    

        splitted_weights        = tf.matmul(weights[:, tf.newaxis], tf.transpose(self._splitting_weights[:,tf.newaxis]))
        levels                  = tf.cast(levels, dtype=tf.int32)
        ews                     = tf.gather(leveled_ews[0], levels, axis=0)
        if directions is None:
            highest_ew_idx          = tf.argmax(ews, axis=-1)
        else:
            highest_ew_idx = directions

        highest_ew              = tf.gather(ews, highest_ew_idx[:,tf.newaxis], axis=-1, batch_dims=1)
        highest_ev              = tf.transpose(tf.gather(orig_ev, highest_ew_idx[:,tf.newaxis], axis=-1)[:,:,:,0])
        square_root_highest_ew  = tf.math.sqrt(highest_ew)
        sq_ew_times_ev          = tf.multiply(square_root_highest_ew, highest_ev[:,:,0])
        splitted_zero_means     = tf.matmul(tf.cast(sq_ew_times_ev[:,:,tf.newaxis], tf.float32), tf.transpose(self._splitting_means[:,tf.newaxis]))
        splitted_means          = tf.linalg.matrix_transpose(tf.add(means[:,:, tf.newaxis], splitted_zero_means))

        splitted_weights_shape  = tf.shape(splitted_weights)
        splitted_means_shape    = tf.shape(splitted_means)

        splitted_weights        = tf.reshape(splitted_weights, (splitted_weights_shape[0]*splitted_weights_shape[1],))
        splitted_means          = tf.reshape(splitted_means, (splitted_means_shape[0]*splitted_means_shape[1], splitted_means_shape[2]))
        splitted_levels         = tf.cast(tf.repeat(levels, self.splitting_level) + 1, dtype=tf.float32)

        return splitted_weights, splitted_means, splitted_levels

    def _burning_stage(self, ew, ev, sample_mean, sample_weight):
        levels                  = tf.map_fn(lambda x: self._get_leveled_ews(x, self.nb_burn), elems=ew)

        splitted_levels         = tf.zeros_like(sample_weight)
        splitted_means          = tf.convert_to_tensor(sample_mean, dtype=tf.float32)
        splitted_weights        = sample_weight

        for _ in range(self.nb_burn):
            splitted_weights, splitted_means, splitted_levels = self._classic_level_split(splitted_weights, splitted_means, splitted_levels, levels, ev)

        return splitted_weights, splitted_means, splitted_levels

    def _fn_split_and_propagate(self, model, weights, means, levels, orig_ew, orig_ev, orig_img_shape):        

        def single_fn_split_and_propagate_condition(out_weights, out_means, out_covs, out_eigenvalues, out_eigenvectors, out_levels, out_criterias, splitted_weights, levels, splitted_means, split_condition, orig_ew=None, orig_ev=None, orig_img_shape=None, model=None):
            return split_condition
        
        split_condition     = True
        out_weights         = tf.convert_to_tensor(np.array([]), dtype=tf.float32)
        out_means           = tf.convert_to_tensor(np.array([]), dtype=tf.float32)
        out_covs            = tf.convert_to_tensor(np.array([]), dtype=tf.float32)
        out_eigenvalues     = tf.convert_to_tensor(np.array([]), dtype=tf.float32)
        out_eigenvectors    = tf.convert_to_tensor(np.array([]), dtype=tf.float32)
        out_levels          = tf.convert_to_tensor(np.array([]), dtype=tf.int32)
        out_criterias       = tf.convert_to_tensor(np.array([]), dtype=tf.float32)

        out_weights, \
            out_means, \
            out_covs, \
            out_eigenvalues, \
            out_eigenvectors, \
            out_levels, \
            out_criterias, \
            weights, \
            levels, \
            means, \
            split_condition = tf.while_loop(
                single_fn_split_and_propagate_condition, 
                lambda x,y,z,z_ew,z_ev,z_l,z_c,a,b,c,d: self._single_sub_fn_split_and_propagate(x, y, z, z_ew, z_ev, z_l, z_c, a, b, c, d, orig_ew, orig_ev, orig_img_shape, model), 
                [out_weights, out_means, out_covs, out_eigenvalues, out_eigenvectors, out_levels, out_criterias, weights, levels, means, split_condition]
                )
            
        return out_weights, out_means, out_covs, out_eigenvalues, out_eigenvectors, out_criterias, out_levels

    def _subspace_fn_propagation2(self, potential_split_weights, level, potential_split_means, leveled_ews, orig_ev, model, orig_img_shape):
        w_points, ut_samples_           = self._subspace_ut_sampling_from_level2(potential_split_weights, level, potential_split_means, leveled_ews, orig_ev)
        ut_sample_shape                 = tf.shape(ut_samples_)

        if len(orig_img_shape) == 3     : ut_samples_ = tf.reshape(ut_samples_, (ut_sample_shape[0]*ut_sample_shape[1], orig_img_shape[0], orig_img_shape[1], orig_img_shape[2]))
        elif len(orig_img_shape) == 2   : ut_samples_ = tf.reshape(ut_samples_, (ut_sample_shape[0]*ut_sample_shape[1], orig_img_shape[0], orig_img_shape[1]))
        elif len(orig_img_shape) == 1   : ut_samples_ = tf.reshape(ut_samples_, (ut_sample_shape[0]*ut_sample_shape[1], orig_img_shape[0]))
        
        linearized_ut_samples_, \
            true_out_ut_samples_        = self._get_linearize_network_outputs(model, ut_samples_)

        # FOR normal network
        linearized_ut_samples           = tf.reshape(linearized_ut_samples_, (ut_sample_shape[0], ut_sample_shape[1], tf.shape(linearized_ut_samples_)[-1]))
        true_out_ut_samples             = tf.reshape(true_out_ut_samples_, (ut_sample_shape[0], ut_sample_shape[1], tf.shape(linearized_ut_samples_)[-1]))

        taylor_criteria                 = tf.square(tf.norm(tf.subtract(linearized_ut_samples, true_out_ut_samples), 2, axis=2))
        taylor_criteria                 = tf.sqrt(tf.reduce_mean(tf.math.multiply(tf.cast(w_points[:,:,0], dtype=tf.float32), taylor_criteria), axis=1))

        linearized_weighted_points      = tf.math.multiply(tf.cast(w_points, dtype=tf.float32), linearized_ut_samples)
        linearized_out_means            = tf.reduce_sum(linearized_weighted_points, axis=1)
        linearized_centered_out_samples = linearized_ut_samples - linearized_out_means[:,tf.newaxis,:]
        linearized_centered_out_samples = linearized_centered_out_samples[:,:,:,tf.newaxis]
        linearized_unweighted_covs      = tf.matmul(linearized_centered_out_samples, tf.linalg.matrix_transpose(linearized_centered_out_samples))
        linearized_weighted_covs        = tf.math.multiply(tf.cast(w_points[:,:,tf.newaxis], dtype=tf.float32), linearized_unweighted_covs)
        linearized_out_covs             = tf.reduce_sum(linearized_weighted_covs, axis=1)

        predicted_weighted_points       = tf.math.multiply(tf.cast(w_points, dtype=tf.float32), true_out_ut_samples)
        predicted_out_means             = tf.reduce_sum(predicted_weighted_points, axis=1)[:,tf.newaxis,:]
        predicted_centered_out_samples  = true_out_ut_samples - predicted_out_means
        predicted_centered_out_samples  = predicted_centered_out_samples[:,:,:,tf.newaxis]
        predicted_unweighted_covs       = tf.matmul(predicted_centered_out_samples, tf.linalg.matrix_transpose(predicted_centered_out_samples))
        predicted_weighted_covs         = tf.math.multiply(tf.cast(w_points[:,:,tf.newaxis], dtype=tf.float32), predicted_unweighted_covs)
        predicted_out_covs              = tf.reduce_sum(predicted_weighted_covs, axis=1)

        return predicted_out_means[:,0,:], predicted_out_covs, linearized_out_means, linearized_out_covs, taylor_criteria

    def _linearization_using_builtin_function(self, model, dataset_x_tf):

        test_dataset    = tf.data.Dataset.from_tensor_slices(dataset_x_tf)
        max_            = np.minimum(self.batch_size, tf.shape(dataset_x_tf)[0])
        test_dataset    = test_dataset.batch(max_, drop_remainder=False)        
        mean            = tf.reduce_mean(dataset_x_tf, axis=0)[tf.newaxis,:,:,:]       

        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(mean)
            output = model(mean)
        mean_pred           = output
        mean_pred_jacobian  = tape.jacobian(output, mean)[0,:,0,:,:,:]
        jac_shape           = tf.shape(mean_pred_jacobian)
        mean_pred_jacobian  = tf.transpose(tf.reshape(mean_pred_jacobian, (jac_shape[0], jac_shape[1]*jac_shape[2]*jac_shape[3])))
        # Taylor linearization: f(x) = f(a) + f'(a)(x-a)        
        data_shape          = tf.shape(dataset_x_tf)
        prediction_bar      = tf.transpose(
            tf.matmul(
                tf.transpose(mean_pred_jacobian), 
                tf.transpose(tf.reshape(dataset_x_tf - mean, (data_shape[0], data_shape[1]*data_shape[2]*data_shape[3])))
                )
            ) + mean_pred
        
        prediction_tilde = None
        for _, batch_test in enumerate(test_dataset):
            cur_prediction_tilde = model(batch_test)            
            if prediction_tilde is None:
                prediction_tilde = cur_prediction_tilde
            else:
                prediction_tilde = tf.concat((prediction_tilde, cur_prediction_tilde), axis=0)               

        return prediction_bar, prediction_tilde

    def _linearization(self, model, dataset_x_tf):
        test_dataset        = tf.data.Dataset.from_tensor_slices(dataset_x_tf)
        max_                = np.minimum(self.batch_size, tf.shape(dataset_x_tf)[0])
        test_dataset        = test_dataset.batch(max_, drop_remainder=False)
        prediction_bar      = None
        prediction_tilde    = None
        for _, batch_test in enumerate(test_dataset):
            cur_prediction_bar, cur_prediction_tilde = model.linear_network(batch_test)            
            if prediction_bar is None:
                prediction_bar      = cur_prediction_bar
                prediction_tilde    = cur_prediction_tilde
            else:
                prediction_bar      = tf.concat((prediction_bar, cur_prediction_bar), axis=0)
                prediction_tilde    = tf.concat((prediction_tilde, cur_prediction_tilde), axis=0)

        return prediction_bar, prediction_tilde

    def _get_linearize_network_outputs(self, model, dataset_x_tf):
        orig_shape = tf.shape(dataset_x_tf)

        if self.is_mlp:
            dataset_x_tf = tf.reshape(dataset_x_tf, (orig_shape[0], orig_shape[1]*orig_shape[2]))

        if self.use_tf_network_linearization:
            prediction_bar, prediction_tilde = self._linearization_using_builtin_function(model, dataset_x_tf)
        else:            
            prediction_bar, prediction_tilde = self._linearization(model, dataset_x_tf)

        return prediction_bar, prediction_tilde

    def _subspace_ut_sampling_from_level2(self, weights, level, means, leveled_ew, orig_ev):
        cur_level               = tf.cast(level, dtype=tf.int32)
        ev_shape                = tf.shape(orig_ev)
        subspace_dim            = ev_shape[2]
        nb_gaussians            = tf.shape(weights)[0]
        ew                      = tf.gather(leveled_ew, cur_level, axis=1)
        subspace_square_root    = tf.linalg.diag(tf.sqrt(ew[0]))

        if subspace_dim == 1    : lambda_ = 2
        elif subspace_dim == 2  : lambda_ = 1
        else                    : lambda_ = subspace_dim.numpy() - 3

        subspace_square_root    = tf.cast(subspace_square_root, tf.float32)
        orig_ev                 = tf.cast(orig_ev, tf.float32)

        subspace_plus_points    = tf.zeros((nb_gaussians, subspace_dim, 1)) + subspace_square_root * np.sqrt(subspace_dim + lambda_)
        subspace_minus_points   = tf.zeros((nb_gaussians, subspace_dim, 1)) - subspace_square_root * np.sqrt(subspace_dim + lambda_)
        zero_point              = tf.zeros((nb_gaussians, 1, subspace_dim))
        subspace_stacked_points = np.hstack([zero_point, tf.linalg.matrix_transpose(subspace_plus_points), tf.linalg.matrix_transpose(subspace_minus_points)])
        upsace_stacked_points   = tf.matmul(subspace_stacked_points, tf.linalg.matrix_transpose(orig_ev)) + means[:,tf.newaxis,:]

        w_points                = tf.ones([nb_gaussians, 2*subspace_dim, 1], dtype=tf.float32) * (1.0/(2.0*(lambda_ + tf.cast(subspace_dim, tf.float32))))
        w0                      = tf.ones([nb_gaussians, 1, 1], dtype=tf.float32) * (lambda_/(lambda_ + tf.cast(subspace_dim, tf.float32)))
        w_points                = tf.concat([w0, w_points], axis=1)
        
        return w_points, upsace_stacked_points

    def _single_sub_fn_split_and_propagate(self, out_weights, out_means, out_covs, out_eigenvalues, out_eigenvectors, out_levels, out_criterias, splitted_weights, splitted_levels, splitted_means, split_condition, orig_ew=None, orig_ev=None, orig_img_shape=None, model=None):

        max_level                       = tf.reduce_max(splitted_levels)
        leveled_ews                     = tf.map_fn(lambda x: self._get_leveled_ews(x, int(max_level)), elems=orig_ew)
        potential_out_means, \
            potential_out_covs, \
            linearized_out_means, \
            linearized_out_covs, \
            taylor_linearization_criteria = self._subspace_fn_propagation2(
                splitted_weights, 
                splitted_levels, 
                splitted_means, 
                leveled_ews, 
                orig_ev, 
                model, 
                orig_img_shape
                )
        
        
        criterias, \
            _, \
            potential_out_eigenvalues, \
            potential_out_eigenvectors = self._non_linearity_detection_criteria_wasserstein( 
                linearized_out_means, 
                linearized_out_covs, 
                potential_out_means, 
                potential_out_covs
                )

        criterias                       = criterias + taylor_linearization_criteria
        weighted_criterias              = tf.multiply(splitted_weights, criterias)

        logger.info(80*"-")
        logger.info(f"Number of criterias {len(weighted_criterias)}")
        logger.info(80*"-")

        mask                        = tf.greater(weighted_criterias, self.splitting_threshold)
        mask_up                     = tf.less(weighted_criterias, self.splitting_threshold)

        apply_boolean_mask          = lambda x, m: tf.boolean_mask(x, m, axis=0)
        concat_tensors              = lambda x, y: tf.concat([x, y], axis=0)
        
        in_tensors                  = [
            splitted_weights, 
            splitted_means,
            splitted_levels,            
            splitted_weights, 
            potential_out_means,
            potential_out_covs, 
            potential_out_eigenvalues, 
            potential_out_eigenvectors, 
            weighted_criterias,
            splitted_levels
            ]
        to_split_weights, \
            to_split_means, \
            to_split_levels, \
            to_split_out_weights, \
            to_split_out_means, \
            to_split_out_covs, \
            to_split_out_eigenvalues, \
            to_split_out_eigenvectors, \
            to_split_out_criterias, \
            to_split_out_levels = list(map(lambda x: apply_boolean_mask(x, mask), in_tensors))
        
        in_tensors                  = [
            splitted_weights, 
            potential_out_means,
            potential_out_covs, 
            potential_out_eigenvalues,
            potential_out_eigenvectors, 
            weighted_criterias,
            splitted_levels
            ]
        cur_out_weights, \
            cur_out_means, \
            cur_out_covs, \
            cur_out_eigenvalues, \
            cur_out_eigenvectors, \
            cur_out_criterias, \
            cur_out_levels          = list(map(lambda x: apply_boolean_mask(x, mask_up), in_tensors))

        nb_out = len(cur_out_weights)
        nb_to_split = len(to_split_weights)
    
        if len(out_weights) > 0 and len(cur_out_weights) > 0:   
            in_tensors = list(zip(
                [out_weights, out_means, out_covs, out_eigenvalues, out_eigenvectors, out_criterias, out_levels], 
                [cur_out_weights, cur_out_means, cur_out_covs, cur_out_eigenvalues, cur_out_eigenvectors, cur_out_criterias, cur_out_levels])
                )

            out_weights, \
                out_means, \
                out_covs, \
                out_eigenvalues, \
                out_eigenvectors, \
                out_criterias, \
                out_levels              = list(map(lambda x: concat_tensors(*x), in_tensors))

        elif len(cur_out_weights) > 0:       
            
            out_weights                 = cur_out_weights
            out_means                   = cur_out_means
            out_covs                    = cur_out_covs
            out_eigenvalues             = cur_out_eigenvalues
            out_eigenvectors            = cur_out_eigenvectors
            out_levels                  = cur_out_levels

        nb_out                      = len(out_weights)
        nb_to_split                 = len(to_split_weights)

        if (nb_out > self.max_split):   
            if nb_to_split > 0:
                
                in_tensors = list(zip(
                    [out_weights, out_means, out_covs, out_eigenvalues, out_eigenvectors, out_criterias, out_levels], 
                    [to_split_out_weights, to_split_out_means, to_split_out_covs, to_split_out_eigenvalues, to_split_out_eigenvectors, to_split_out_criterias, to_split_out_levels])
                    )

                out_weights, \
                    out_means, \
                    out_covs, \
                    out_eigenvalues, \
                    out_eigenvectors, \
                    out_criterias, \
                    out_levels              = list(map(lambda x: concat_tensors(*x), in_tensors))

            splitted_weights    = tf.zeros(shape=[0], dtype=tf.float32)
            splitted_means      = tf.zeros(shape=[0], dtype=tf.float32)
            splitted_levels     = tf.zeros(shape=[0], dtype=tf.float32)
            # splitted_ew         = tf.zeros(shape=[0], dtype=tf.float32)
            # splitted_ev         = tf.zeros(shape=[0], dtype=tf.float32)
            split_condition     = False
            
        elif nb_out + 7*nb_to_split >= self.max_split:  
            nb_split                    = (self.max_split - nb_out - nb_to_split) // 6

            in_tensors                  = [
                to_split_out_weights, 
                to_split_out_means, 
                to_split_out_covs, 
                to_split_out_eigenvalues, 
                to_split_out_eigenvectors, 
                to_split_out_criterias, 
                to_split_out_levels
                ]
            to_keep_weights, \
                to_keep_means, \
                to_keep_covs, \
                to_keep_ews, \
                to_keep_evs, \
                to_keep_criterias, \
                to_keep_levels = list(map(lambda x: x[nb_split:], in_tensors)) 
            
            to_split_weights = to_split_weights[:nb_split]
            to_split_means = to_split_means[:nb_split]
            to_split_levels = to_split_levels[:nb_split]

            splitted_weights, \
                splitted_means, \
                splitted_levels     = self._classic_level_split(
                    to_split_weights, 
                    to_split_means, 
                    to_split_levels, 
                    leveled_ews, 
                    orig_ev)

            if len(out_means) > 0:
                
                in_tensors = list(zip(
                    [out_weights, out_means, out_covs, out_eigenvalues, out_eigenvectors, out_criterias, out_levels], 
                    [to_keep_weights, to_keep_means, to_keep_covs, to_keep_ews, to_keep_evs, to_keep_criterias, to_keep_levels])
                    )
                
                out_weights, \
                    out_means, \
                    out_covs, \
                    out_eigenvalues, \
                    out_eigenvectors, \
                    out_criterias, \
                    out_levels              = list(map(lambda x: concat_tensors(*x), in_tensors))                
            else:  
                out_weights         = to_keep_weights
                out_means           = to_keep_means
                out_covs            = to_keep_covs
                out_eigenvalues     = to_keep_ews
                out_eigenvectors    = to_keep_evs
                out_criterias       = to_keep_criterias
                out_levels          = to_keep_levels

            if nb_split > 0: split_condition = True
            else           : split_condition = False 

        elif nb_to_split > 0:   
            splitted_weights, \
                splitted_means, \
                splitted_levels = self._classic_level_split(
                    to_split_weights, 
                    to_split_means, 
                    to_split_levels, 
                    leveled_ews, 
                    orig_ev
                    )
            split_condition = True
        else:
            split_condition = False  

        return out_weights, out_means, out_covs, out_eigenvalues, out_eigenvectors, out_levels, out_criterias, splitted_weights, splitted_levels, splitted_means, split_condition                          
        
    def _non_linearity_detection_criteria_wasserstein(self, means, covs, out_means, out_covs):
        distances, \
            directions, \
            out_ew, \
            out_ev      = self._calculate_wasserstein_distance_for_splitting(out_means, out_covs, means, covs)
        
        return distances, directions, out_ew, out_ev

    def _calculate_wasserstein_distance_for_splitting(self, mean2, cov2, mean1, cov1):
        diff_mean            = tf.subtract(mean1, mean2)
        mean_norm            = tf.square(tf.linalg.norm(diff_mean, 2, axis=1))
        ew2, \
            ev2, \
            square_root_cov2 = self._matrix_square_root(cov2, 1e-5)
        cov21                = tf.matmul(square_root_cov2, tf.matmul(cov1, square_root_cov2))
        _, _, cov21_sqrt     = self._matrix_square_root(cov21, 1e-5)
        cov21_sqrt           = -2.0*cov21_sqrt
        final_mat            = tf.add(cov1, tf.add(cov2, cov21_sqrt))
        tr                   = tf.linalg.trace(final_mat)
        tr                   = tf.where(tf.less(tr, 0.0), 0.0, tr)

        wasserstein_distance = tf.sqrt(mean_norm + tr)

        return wasserstein_distance, None, ew2, ev2

    def _matrix_square_root(self, covs, projection_level=1e-5):
        ew, ev          = tf.linalg.eigh(covs)
        sqrt_covs       = self._matrix_square_root_by_decomposition(ew, ev, projection_level)

        return ew, ev, sqrt_covs       

    def _matrix_square_root_by_decomposition(self, ew, ev, projection_level=1e-5):
        clip_ew         = tf.where(tf.less(ew, projection_level), projection_level, ew)
        sqrt_clip_ew    = tf.sqrt(clip_ew)
        sqrt_covs       = tf.matmul(ev, tf.matmul(tf.linalg.diag(sqrt_clip_ew), tf.linalg.matrix_transpose(ev)))

        return sqrt_covs             

    def propagate(self, img, model, sample_mean, sample_cov, sample_weight):
        logger.info(f"call gmm_ut_w_fn_propagation")
        logger.info(f"sample covariance shape {sample_cov.shape}")

        wgmpropresults_obj      = WGMpropResults()   

        sample_weight           = tf.convert_to_tensor(sample_weight, dtype=tf.float32)
        
        if self.use_subspace:
            ew, ev                  = self._compute_eigen_components_using_subspace(sample_cov, sample_weight) #tf.map_fn(lambda x: self._compute_eigen_components_using_subspace(x[0], x[1]), elems=(sample_cov, sample_weight)) #
        else:
            ew, ev                  = self._compute_eigen_components(sample_cov, sample_weight)
        
        splitted_weights, \
            splitted_means, \
            splitted_levels     = self._burning_stage(ew, ev, sample_mean, sample_weight)
        gmm_out_weights, \
            gmm_out_means, \
            gmm_out_vars, \
            _, _, \
            out_criterias, \
            levels              = self._fn_split_and_propagate(model, splitted_weights, splitted_means, splitted_levels, ew, ev, orig_img_shape=tf.shape(img))

        final_criteria          = tf.reduce_sum(out_criterias)

        wgmpropresults_obj.gmm_out_weights  = gmm_out_weights
        wgmpropresults_obj.gmm_out_means    = gmm_out_means
        wgmpropresults_obj.gmm_out_vars     = gmm_out_vars
        wgmpropresults_obj.levels           = levels
        wgmpropresults_obj.final_criteria   = final_criteria
        wgmpropresults_obj.ew               = ew
        wgmpropresults_obj.method           = PropagationMethod.GMM_UT_W_FN

        return wgmpropresults_obj
