import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import logging
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from model.utils_experiments import study_sample_updated
from model.utils_user_inputs import ModelLoader
from model.utils_user_inputs import UserFileProcessor
from model.utils_dataset import InputPreprocessor
from model.utils_dataset import DatasetLoader
from model.utils_dataset import get_input_shape
from model.utils_dataset import get_test_cases
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

def main(filename):
    userfileprocessor_obj   = UserFileProcessor(filename=filename)
    parameters, \
        noises, \
            intensities     = userfileprocessor_obj.process()
    
    np.random.seed(parameters.seed)
    tf.random.set_seed(parameters.seed)
    random.seed(parameters.seed)
    
    datasetloader_obj       = DatasetLoader(dataset_name=parameters.dataset)
    x_test, y_test          = datasetloader_obj.load_test_dataset()
    inputpreprocessor_obj   = InputPreprocessor(model_name=parameters.model_type)
    x_test, y_test          = inputpreprocessor_obj.preprocess(x_test=x_test, y_test=y_test)
    input_shape             = get_input_shape(x_test=x_test)
    modelloader_obj         = ModelLoader(parameters=parameters, input_shape=input_shape)
    model                   = modelloader_obj.get_built_model()
    test_cases              = get_test_cases(intensities, noises, nb_test=parameters.nb_test, max=len(x_test))
    is_mlp                  = (parameters.model_type == 'mlp')

    
    
    for intensity, noise, i in test_cases:
        parameters.noise_type      = noise
        parameters.noise_intensity = intensity
        parameters.image_prefix    = os.path.join(parameters.image_prefix_root, f"{noise.name}_{intensity.name}")      

        study_sample_updated(model=model, cur_image=x_test[i], cur_label=y_test[i], index=i, parameters=parameters, is_mlp=is_mlp) 

if __name__=="__main__":
    main(filename="run_param_mnist.json")
