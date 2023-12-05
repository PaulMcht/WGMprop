import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as uvn
import ot
import scipy.sparse as scsp
import scipy as sc
import logging
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

#############################################################################################################################################################
#                                                                                                                                                           #
#                                                                       UTILS                                                                               #
#                                                                                                                                                           #
#############################################################################################################################################################

def compute_percentile_from_samples(samples, percentile):
    if percentile < 1:
        percentile = percentile * 100
    
    return np.percentile(samples, percentile)

#############################################################################################################################################################
#                                                                                                                                                           #
#                                                            PROJECTION TO SN++(R)                                                                          #
#                                                                                                                                                           #
#############################################################################################################################################################

def covariance_matrix_projection_by_addition(covs, projection_level=1e-5):
    cov_shape = tf.shape(covs)
    projected_covs = covs + projection_level * tf.eye(cov_shape[1], cov_shape[2], [cov_shape[0]])

    return projected_covs

def covariance_matrix_projection(covs, projection_level=1e-5):
    ew, ev = tf.linalg.eigh(covs)
    projected_covs = covariance_matrix_projection_by_decomposition(ew, ev, projection_level)

    return ew, ev, projected_covs

def covariance_matrix_projection_by_decomposition(ew, ev, projection_level=1e-5):
    clip_ew = tf.where(tf.less(ew, projection_level), projection_level, ew)
    projected_covs = tf.matmul(ev, tf.matmul(tf.linalg.diag(clip_ew), tf.linalg.matrix_transpose(ev)))

    return projected_covs

def matrix_square_root(covs, projection_level=1e-5):
    ew, ev = tf.linalg.eigh(covs)
    sqrt_covs = matrix_square_root_by_decomposition(ew, ev, projection_level)

    return ew, ev, sqrt_covs

def matrix_square_root_by_decomposition(ew, ev, projection_level=1e-5):
    clip_ew = tf.where(tf.less(ew, projection_level), projection_level, ew)
    sqrt_clip_ew = tf.sqrt(clip_ew)
    sqrt_covs = tf.matmul(ev, tf.matmul(tf.linalg.diag(sqrt_clip_ew), tf.linalg.matrix_transpose(ev)))

    return sqrt_covs

def matrix_inverse(covs, projection_level=1e-5):
    ew, ev = tf.linalg.eigh(covs)
    inv_covs = matrix_inverse_by_decomposition(ew, ev, projection_level=projection_level)

    return ew, ev, inv_covs

def matrix_inverse_by_decomposition(ew, ev, projection_level=1e-5):
    rank = tf.reduce_sum(tf.where(tf.greater(ew, projection_level), 1, 0), axis=1)
    inv_covs, _, _ = tf.map_fn(lambda x: single_matrix_inverse(x[0], x[1], x[2]), elems=(ev, ew, rank))

    return inv_covs

def single_matrix_inverse(ev, ew, rank):
    Q_r = ev[:,-rank:]
    positive_ew = ew[-rank:]
    inv_clip_ew = tf.divide(1.0, positive_ew)

    inv_cov = tf.matmul(Q_r, tf.matmul(tf.linalg.diag(inv_clip_ew), tf.linalg.matrix_transpose(Q_r)))

    return inv_cov, ew, rank

def sparse_eigh_testing(cov, useless, k=10, ncv=100):
    '''
    Compute the k highest eigenvalues of cov using scipy sparse
    '''
    return scsp.linalg.eigsh(cov, k, which='LM', ncv=ncv)

def sparse_eigh(cov, useless, k=10, ncv=100):
    '''
    Compute the k highest eigenvalues of cov using scipy sparse
    '''
    return scsp.linalg.eigsh(cov, k, which='LM')
    # return scsp.linalg.eigsh(cov, k, which='LM', ncv=ncv)

def eigh(cov, useless):
    return sc.linalg.eigh(cov.numpy())

#############################################################################################################################################################
#                                                                                                                                                           #
#                                                           KL DIVERGENCE COMPUTING                                                                         #
#                                                                                                                                                           #
#############################################################################################################################################################

def compute_probs(data, n=None):
    if n is not None:
        n_bins = n
    else:
        IQR = np.percentile(data, 75) - np.percentile(data, 25)
        bin_width = 2.0*IQR/(len(data)**(1/3))
        max_ = np.percentile(data, 99)
        min_ = np.percentile(data, 1)
        if bin_width > 0.001:
            n_bins = int((max_ - min_)/bin_width)
        else:
            n_bins = 100

    h, e = np.histogram(data, n_bins)
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q):
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int

def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q):
    tmp = p*np.log(p/q)
    return np.sum(tmp)

def compute_kl_divergence(train_sample, test_sample, n_bins=10000):
    """
    Computes the KL Divergence using the support
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=None)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)

    return kl_divergence(p, q)

#############################################################################################################################################################
#                                                                                                                                                           #
#                                                                   SAMPLING                                                                                #
#                                                                                                                                                           #
#############################################################################################################################################################

def sample_from_gmm_mvn(weights, means, covs, nb=10**7):
    """Sample from the gaussian mixture model specified in parameters

    Args:
        weights (array): the weight of the gmm components
        means (array): the mean of the gmm components
        covs (array): the cov matrix of the gmm components
        nb (int, optional): The number of samples. Defaults to 10**7.

    Returns:
        array: the array containing the nb samples from the input 
    """    
    million_samples = np.random.multinomial(nb, weights, size=1)

    if np.sum(million_samples) != nb:
        million_samples[0,0] = nb - np.sum(million_samples) + million_samples[0,0]

    gmm_samples = None
    for i in range(million_samples.shape[1]):
        nb_sample = million_samples[0,i]

        if nb_sample > 0:
            ew, ev = np.linalg.eigh(covs[i])
            ew[ew < 1e-6] = 1e-6
            cur_cov = ev @ np.diag(ew) @ np.transpose(ev)

            cur_samples = np.array(mvn(means[i], cur_cov).rvs(million_samples[0,i]))

            if len(cur_samples.shape) == 1:
                cur_samples = cur_samples[np.newaxis,:]

            if gmm_samples is None:
                gmm_samples = cur_samples
            else:
                gmm_samples = np.vstack([gmm_samples, cur_samples])
    

    return gmm_samples

def sample_from_gmm_uvn(weights, means, covs, nb=10**7):
    """Sample from the gaussian mixture model specified in parameters

    Args:
        weights (array): the weight of the gmm components
        means (array): the mean of the gmm components
        covs (array): the cov matrix of the gmm components
        nb (int, optional): The number of samples. Defaults to 10**7.

    Returns:
        array: the array containing the nb samples from the input 
    """    
    million_samples = np.random.multinomial(nb, weights, size=1)

    if np.sum(million_samples) != nb:
        million_samples[0,0] = nb - np.sum(million_samples) + million_samples[0,0]

    gmm_samples = None
    for i in range(million_samples.shape[1]):
        nb_sample = million_samples[0,i]

        if nb_sample > 0:
            ew, ev = np.linalg.eigh(covs[i])
            ew[ew < 1e-6] = 1e-6
            cur_cov = ev @ np.diag(ew) @ np.transpose(ev)

            cur_samples = np.array(uvn(means[i], cur_cov).rvs(million_samples[0,i]))
            if nb_sample == 1:
                cur_samples = np.array([cur_samples])

            if len(cur_samples.shape) == 1:
                cur_samples = cur_samples[:,np.newaxis]

            if gmm_samples is None:
                gmm_samples = cur_samples
            else:
                gmm_samples = np.vstack([gmm_samples, cur_samples])
    

    return gmm_samples

#############################################################################################################################################################
#                                                                                                                                                           #
#                                                    WASSERSTEIN DISTANCE COMPUTING                                                                         #
#                                                                                                                                                           #
#############################################################################################################################################################

def compute_wasserstein_distance(u, v, p=2):
    """Compute the p-wasserstein distance using POT library

    Args:
        u (list): list of samples of first distribution
        v (list): list of samples of second distribution
        p (int, optional): the p order of the wasserstein distance. Defaults to 2.

    Returns:
        _type_: _description_
    """    
    assert len(u) == len(v)
    return ot.wasserstein_1d(u, v, u_weights=None, v_weights=None, p=p, require_sort=True)

#############################################################################################################################################################
#                                                                                                                                                           #
#                                                    MISCELLEANOUS                                                                         #
#                                                                                                                                                           #
#############################################################################################################################################################

def moment_estimation_from_samples(flatten_images_, use_approx=True):
    logger.info(f"call moment_estimation_from_samples")

    # UPDATED CODE (in progress)  
    if use_approx:
        pca_full    = PCA(n_components=10, svd_solver='full')
        pca_full.fit(flatten_images_)
        gmm_var     = pca_full.get_covariance()
        gmm_mean    = np.mean(flatten_images_, axis=0)[np.newaxis,:].astype(np.float32)
        del pca_full
    else:
        # ORIGINAL CODE 
        gmm_mean = np.mean(flatten_images_, axis=0, dtype=np.float32)[np.newaxis,:]
        gmm_var = np.cov(flatten_images_.T, dtype=np.float32)#[np.newaxis,:,:]
    
    return gmm_mean, gmm_var

def compute_moments_from_mixture(weights, means, covs):
    logger.info("call compute_moments_from_mixture")

    out_mean = tf.reduce_sum(tf.multiply(weights[:,tf.newaxis], means), axis=0)
    weighted_out_vars = tf.multiply(weights[:,tf.newaxis,tf.newaxis], covs)
    centered_out_means = tf.subtract(means, out_mean)[:,:,tf.newaxis]
    out_var = tf.reduce_sum(tf.add(weighted_out_vars, tf.multiply(weights[:,tf.newaxis, tf.newaxis], tf.matmul(centered_out_means, tf.linalg.matrix_transpose(centered_out_means)))), axis=0)

    del weighted_out_vars, centered_out_means
    return out_mean, out_var
