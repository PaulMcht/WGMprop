import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import logging
import logging.config

from model.utils import sample_from_gmm_uvn 
from model.utils import sample_from_gmm_mvn
from model.utils import compute_kl_divergence
from model.utils import compute_percentile_from_samples
from model.utils import compute_wasserstein_distance
from model.utils import moment_estimation_from_samples
from model.utils import compute_moments_from_mixture
from model.enums import PropagationMethod
from model.utils_noise import NoiseGenerator
from model.utils_wgmprop_propagation import WGMpropPropagation
from model.utils_mc_propagation import MonteCarloPropagation

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ExecutionTimeResults:
    def __init__(self) -> None:
        self.gen_time = None
        self.mc_time  = None
        self.gmm_time = None

class AnalyseResults:
    def __init__(self) -> None:
        self.gmm_out_mean       = None 
        self.gmm_out_cov        = None 
        self.kl_div             = None 
        self.wasserstein_dst    = None 
        self.mc_percentiles     = None 
        self.gmm_percentiles    = None

def study_sample_updated(model, cur_image, cur_label, index, parameters, is_mlp=False):
    
    logger.info(f"Study sample number {index}")
    logger.info(f"Sample image shape {cur_image.shape}")

    executiontimeresults_obj    = ExecutionTimeResults()
    
    logger.info(f"Generation of noisy samples")
    noisegenerator_obj          = NoiseGenerator(noise_type=parameters.noise_type, noise_intensity=parameters.noise_intensity, nb_noisy_samples=parameters.nb_noisy_sample)
    gen_tic                     = time.time()
    flatten_images_, _          = noisegenerator_obj.generate(source_image=cur_image, cur_label=cur_label)
    executiontimeresults_obj.gen_time  = time.time()-gen_tic
    
    logger.info(f"Estimation of moments using the generated samples")
    sample_mean, sample_cov     = moment_estimation_from_samples(flatten_images_, use_approx=False)

    if parameters.with_ref:
        logger.info(f"Building Monte Carlo reference using {flatten_images_.shape[0]-1} samples")
        mc_tic                      = time.time()
        montecarlopropagation_obj   = MonteCarloPropagation(**vars(parameters), is_mlp=is_mlp)
        mc_results                  = montecarlopropagation_obj.propagate(img=cur_image, model=model, sample_mean=sample_mean, sample_cov=sample_cov, sample_weight=np.array([1.0]))
        executiontimeresults_obj.mc_time   = time.time()-mc_tic

    if parameters.method_name == PropagationMethod.GMM_UT_W_FN.name:
        logger.info(f"Propagation of uncertainty using {parameters.method_name} method")
        gmm_tic                     = time.time()
        wgmproppropagation_obj      = WGMpropPropagation(**vars(parameters), is_mlp=is_mlp)
        wgmprop_results             = wgmproppropagation_obj.propagate(img=cur_image, model=model, sample_mean=sample_mean, sample_cov=sample_cov, sample_weight=np.array([1.0]))
        executiontimeresults_obj.gmm_time  = time.time()-gmm_tic
        prefix_             = parameters.image_prefix + "_" + PropagationMethod.GMM_UT_W_FN.name + "_"   

    if parameters.visualize:
        logger.info(f"Save figures of marginals")            
        show_all_methods_marginals_(index, wgmprop_results, mc_results, prefix=prefix_, nb_class=parameters.nb_class)
        
    if parameters.save_results:        
        if parameters.with_ref:
            logger.info(f"Analyze the results")
            analyse_results = analyse_method(mc_results, wgmprop_results, parameters=parameters)
            logger.info(f"Save results in a .json file")
            save_result_in_json(analyse_results, wgmprop_results, mc_results, executiontimeresults_obj, method=wgmprop_results.method, index=index, label=cur_label, parameters=parameters)

    return 

#################################################################
#                                                               #
#                       RESULTS SECTION                         #
#                                                               #
#################################################################
def analyse_method(mc_results, wgmprop_results, parameters, *args, **kwargs):
    if tf.is_tensor(mc_results.predictions):
        mc_predictions = mc_results.predictions.numpy()
    else:
        mc_predictions = mc_results.predictions
    if tf.is_tensor(wgmprop_results.gmm_out_means):
        gmm_out_means = wgmprop_results.gmm_out_means.numpy()
    else:
        gmm_out_means = wgmprop_results.gmm_out_means
    if tf.is_tensor(wgmprop_results.gmm_out_weights):
        gmm_out_weights = wgmprop_results.gmm_out_weights.numpy()
    else:
        gmm_out_weights = wgmprop_results.gmm_out_weights
    if tf.is_tensor(wgmprop_results.gmm_out_vars):
        gmm_out_vars = wgmprop_results.gmm_out_vars.numpy()
    else:
        gmm_out_vars = wgmprop_results.gmm_out_vars
    ###########################################################################################################
    #                                   GMM MOMENTS CALCULATION                                               #
    ###########################################################################################################
    gmm_out_mean, gmm_out_cov = compute_moments_from_mixture(gmm_out_weights, gmm_out_means, gmm_out_vars)
    ###########################################################################################################
    #                                            KL DIV                                                       #
    ###########################################################################################################
    if parameters.nb_class == 1:
        gmm_samples = sample_from_gmm_uvn(gmm_out_weights, gmm_out_means, gmm_out_vars, nb=parameters.ref_nb_sample)
    else:
        gmm_samples = sample_from_gmm_mvn(gmm_out_weights, gmm_out_means, gmm_out_vars, nb=parameters.ref_nb_sample)

    kl_div = []

    argsort = np.argsort(np.mean(mc_predictions, axis=0))

    if mc_predictions.shape[1] > 10:
        nb_marg = 10
    else:
        nb_marg = mc_predictions.shape[1]

    for i in range(nb_marg):
        i = i + 1
        cur_index = argsort[-i]
        kl_div.append(compute_kl_divergence(mc_predictions[:,cur_index], gmm_samples[:,cur_index], n_bins=parameters.kl_bins))

    wasserstein_dst = []
    for i in range(nb_marg):
        i = i + 1
        cur_index = argsort[-i]
        cur_wass_dst = compute_wasserstein_distance(mc_predictions[:,cur_index], gmm_samples[:,cur_index])
        wasserstein_dst.append(cur_wass_dst)
        
    ###########################################################################################################
    #                                            PERCENTILES                                                  #
    ###########################################################################################################
    percentiles     = parameters.percentiles
    mc_percentiles  = []
    gmm_percentiles = []

    for i in range(nb_marg):
        cur_index           = argsort[-i]
        cur_mc_percentiles  = []
        cur_gmm_percentiles = []

        for percentile in percentiles:
            mc_percentile  = compute_percentile_from_samples(gmm_samples[:,cur_index], percentile)
            gmm_percentile = compute_percentile_from_samples(mc_predictions[:,cur_index], percentile)

            cur_mc_percentiles.append(mc_percentile)
            cur_gmm_percentiles.append(gmm_percentile)

        mc_percentiles.append(cur_mc_percentiles)
        gmm_percentiles.append(cur_gmm_percentiles)

    analyseresults_obj                  = AnalyseResults()
    analyseresults_obj.gmm_out_mean     = gmm_out_mean
    analyseresults_obj.gmm_out_cov      = gmm_out_cov
    analyseresults_obj.kl_div           = kl_div
    analyseresults_obj.wasserstein_dst  = wasserstein_dst
    analyseresults_obj.mc_percentiles   = mc_percentiles
    analyseresults_obj.gmm_percentiles  = gmm_percentiles

    return analyseresults_obj

def show_all_methods_marginals_(index, wgmprop_results, mc_results, prefix="_camelyon_index_", nb_class=2):
        mc_predictions = mc_results.predictions
        method_out_weights = wgmprop_results.gmm_out_weights
        method_out_means = wgmprop_results.gmm_out_means
        method_out_vars = wgmprop_results.gmm_out_vars
        
        mc_min = np.amin(mc_predictions)
        mc_max = np.amax(mc_predictions)
        x_axis = np.arange(0.9*mc_min, 1.1*mc_max, (1.1*mc_max - 0.9*mc_min)/10000.0)

        for j in range(nb_class):
            plt.figure()
            overall_pdf = None
            if mc_predictions is not None:
                x0 = mc_predictions[:,j]
                IQR = np.percentile(x0, 75) - np.percentile(x0, 25)
                bin_width = 2.0*IQR/(len(x0)**(1/3))
                max_ = np.percentile(x0, 99)
                min_ = np.percentile(x0, 1)
                if bin_width > 0.001:
                    n_bins = int((max_ - min_)/bin_width)
                else:
                    n_bins = 100
                
                n, bins = np.histogram(x0, bins=n_bins, density=True)
                sns.histplot(x0, kde=False, bins=n_bins, stat="density", color="silver", label="MC")
                plt.title("Output PDF Comparison, image n°{}, marginale n°{}".format(index, j))
                plt.xlabel("Prediction")
                plt.ylabel("Normalized output probability density function")

            
            for i in range(len(method_out_weights)):
                cur_mean = np.array(method_out_means)[i,j]
                cur_cov = np.array(method_out_vars)[i,j,j]
          
                if cur_cov < 1e-10:
                    cur_cov = 1e-10
                cur_weight = np.array(method_out_weights)[i]
                cur_std = np.sqrt(cur_cov)
                

                cur_pdf = cur_weight*(1/(np.sqrt(2*np.pi)*cur_std))*np.exp(-0.5*((x_axis-cur_mean)**2)/(cur_cov))
                
                if overall_pdf is None:
                    overall_pdf = cur_pdf
                else:
                    overall_pdf = overall_pdf + cur_pdf

            plt.plot(x_axis, overall_pdf, label="WGMprop", linewidth=1.5)
            # plt.xlim(left=900)
            # plt.xlim(right=1500)
            # plt.ylim(top=0.015)
            # plt.ylim(bottom=0.0)
            plt.legend()
            plt.savefig(prefix + str(index) + '_marg_' + str(j) + '.png')
            plt.close()
        return

def save_result_in_json(analyse_results, wgmprop_results, mc_results, executiontimeresults_obj, method, index, label, parameters):

    if tf.is_tensor(label):
        label = label.numpy().tolist()
    elif isinstance(label, np.ndarray):
        label = label.tolist()
    else: 
        int(label)

    if tf.is_tensor(wgmprop_results.gmm_out_weights):
        gmm_out_weights = wgmprop_results.gmm_out_weights.numpy().tolist()
    elif isinstance(wgmprop_results.gmm_out_weights, np.ndarray):
        gmm_out_weights = wgmprop_results.gmm_out_weights.tolist()
    else:
        gmm_out_weights = wgmprop_results.gmm_out_weights
    
    if tf.is_tensor(wgmprop_results.gmm_out_means):
        gmm_out_means = wgmprop_results.gmm_out_means.numpy().tolist()
    elif isinstance(wgmprop_results.gmm_out_means, np.ndarray):
        gmm_out_means = wgmprop_results.gmm_out_means.tolist()
    else:
        gmm_out_means = wgmprop_results.gmm_out_means
    
    if tf.is_tensor(wgmprop_results.gmm_out_vars):
        gmm_out_vars = wgmprop_results.gmm_out_vars.numpy().tolist()
    elif isinstance(wgmprop_results.gmm_out_vars, np.ndarray):
        gmm_out_vars = wgmprop_results.gmm_out_vars.tolist()
    else:
        gmm_out_vars = wgmprop_results.gmm_out_vars
    
    if tf.is_tensor(wgmprop_results.levels):
        levels = wgmprop_results.levels.numpy().tolist()
    elif isinstance(wgmprop_results.levels, np.ndarray):
        label = wgmprop_results.levels.tolist()
    else:
        levels = wgmprop_results.levels
    
    if tf.is_tensor(wgmprop_results.final_criteria):
        final_criteria = wgmprop_results.final_criteria.numpy().tolist()
    elif isinstance(wgmprop_results.final_criteria, np.ndarray):
        final_criteria = wgmprop_results.final_criteria.tolist()
    else:
        final_criteria = wgmprop_results.final_criteria
    
    if tf.is_tensor(wgmprop_results.ew):
        ew = wgmprop_results.ew.numpy().tolist()
    elif isinstance(wgmprop_results.ew, np.ndarray):
        ew = wgmprop_results.ew.tolist()
    else:
        ew = wgmprop_results.ew
    
    if tf.is_tensor(executiontimeresults_obj.gmm_time):
        gmm_time = executiontimeresults_obj.gmm_time.numpy()
    elif isinstance(executiontimeresults_obj.gmm_time, np.ndarray):
        gmm_time = executiontimeresults_obj.gmm_time.tolist()
    else:
        gmm_time = executiontimeresults_obj.gmm_time
    
    if tf.is_tensor(executiontimeresults_obj.mc_time):
        mc_time = executiontimeresults_obj.mc_time.numpy()
    elif isinstance(executiontimeresults_obj.mc_time, np.ndarray):
        mc_time = executiontimeresults_obj.mc_time.tolist()
    else:
        mc_time = executiontimeresults_obj.mc_time
        
    if tf.is_tensor(executiontimeresults_obj.gen_time):
        gen_time = executiontimeresults_obj.gen_time.numpy()
    elif isinstance(executiontimeresults_obj.gen_time, np.ndarray):
        gen_time = executiontimeresults_obj.gen_time.tolist()
    else:
        gen_time = executiontimeresults_obj.gen_time
    
    if tf.is_tensor(analyse_results.gmm_percentiles):
        gmm_percentiles = analyse_results.gmm_percentiles.numpy().tolist()
    elif isinstance(analyse_results.gmm_percentiles, np.ndarray):
        gmm_percentiles = analyse_results.gmm_percentiles.tolist()
    else:
        gmm_percentiles = analyse_results.gmm_percentiles

    if tf.is_tensor(analyse_results.mc_percentiles):
        mc_percentiles = analyse_results.mc_percentiles.numpy().tolist()
    elif isinstance(analyse_results.mc_percentiles, np.ndarray):
        mc_percentiles = analyse_results.mc_percentiles.tolist()
    else:
        mc_percentiles = analyse_results.mc_percentiles
        
    if tf.is_tensor(analyse_results.wasserstein_dst):
        wasserstein_dst = analyse_results.wasserstein_dst.numpy().tolist()
    elif isinstance(analyse_results.wasserstein_dst, np.ndarray):
        wasserstein_dst = analyse_results.wasserstein_dst.tolist()
    else:
        wasserstein_dst = analyse_results.wasserstein_dst
    
    if tf.is_tensor(analyse_results.kl_div):
        kl_div = analyse_results.kl_div.numpy().tolist()
    elif isinstance(analyse_results.kl_div, np.ndarray):
        kl_div = analyse_results.kl_div.tolist()
    else:
        kl_div = analyse_results.kl_div
        
    if tf.is_tensor(analyse_results.gmm_out_cov):
        gmm_out_cov = analyse_results.gmm_out_cov.numpy().tolist()
    elif isinstance(analyse_results.gmm_out_cov, np.ndarray):
        gmm_out_cov = analyse_results.gmm_out_cov.tolist()
    else:
        gmm_out_cov = analyse_results.gmm_out_cov
        
    if tf.is_tensor(analyse_results.gmm_out_mean):
        gmm_out_mean = analyse_results.gmm_out_mean.numpy().tolist()
    elif isinstance(analyse_results.gmm_out_mean, np.ndarray):
        gmm_out_mean = analyse_results.gmm_out_mean.tolist()
    else:
        gmm_out_mean = analyse_results.gmm_out_mean
        
    if tf.is_tensor(mc_results.predictions):
        mc_predictions = mc_results.predictions.numpy().tolist()
    elif isinstance(mc_results.predictions, np.ndarray):
        mc_predictions = mc_results.predictions.tolist()
    else:
        mc_predictions = mc_results.predictions
        
    if tf.is_tensor(mc_results.cov):
        mc_out_cov = mc_results.cov.numpy().tolist()
    elif isinstance(mc_results.cov, np.ndarray):
        mc_out_cov = mc_results.cov.tolist()
    else:
        mc_out_cov = mc_results.cov
        
    if tf.is_tensor(mc_results.mean):
        mc_out_mean = mc_results.mean.numpy().tolist()
    elif isinstance(mc_results.mean, np.ndarray):
        mc_out_mean = mc_results.mean.tolist()
    else:
        mc_out_mean = mc_results.mean

    nb_gaussian = len(gmm_out_weights)

    metadata_dict = { "metadata": {
        "split_threhold": parameters.splitting_threshold,
        "splitting_level": parameters.splitting_level,
        "nb_mc_samples": parameters.ref_nb_sample,
        "nb_noisy_samples": parameters.nb_noisy_sample,
        "noise_type": parameters.noise_type.name,
        "noise_intensity": parameters.noise_intensity.name,
        "weight_filename": parameters.model_path,
        "percentiles": parameters.percentiles
        }
    }

    result_dict = {index:{
        "results": {
            "index": index,
            "kl": kl_div,
            "wasserstein": wasserstein_dst,
            "gen_time": gen_time,
            "final_criteria": final_criteria,
            "mc": {
                    "mc_time": mc_time,
                    "out_mean": mc_out_mean,
                    "out_cov": mc_out_cov,
                    "percentiles": mc_percentiles
                    },
            "propagation": {
                    "propagation_method": method.name,
                    "nb": nb_gaussian,
                    "levels": levels,
                    "ew": ew,
                    "gmm_time": gmm_time,
                    "out_mean": gmm_out_mean,
                    "out_cov": gmm_out_cov,
                    "percentiles": gmm_percentiles
                    }
        }
    }}

    result_filename = os.path.join('logs', parameters.dataset, f"{method.name}_{parameters.noise_type.name}_{parameters.noise_intensity.name}.json")
    result_path     = os.path.join(os.getcwd(), result_filename)

    if os.path.exists(result_path):
        with open(result_path) as json_file:
            data = json.load(json_file)
            data.update(result_dict)
    else:
        data = result_dict
        data.update(dict(metadata_dict))
 
    with open(result_path, "w") as write_file:
        json.dump(data, write_file)

    return
