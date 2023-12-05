import os
import json
from types import SimpleNamespace
from typing import Tuple
import logging
from model.enums import PropagationMethod
from model.enums import NoiseIntensity
from model.enums import NoiseType
from model.enums import ModelType

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ModelLoader:
    def __init__(self, parameters, input_shape):
        self.parameters     = parameters
        self.input_shape    = input_shape
        self.model_type     = ModelType(parameters.model_type.lower())
        self.model_funcs    = {
            ModelType.MLP       : self.get_mlp_model,
            ModelType.VGG_CIFAR10       : self.get_vgg_cifar10_model,
            ModelType.VGG_CIFAR100       : self.get_vgg_cifar100_model
        }
        self.weights_path_funcs = {
            ModelType.MLP       : self.get_mlp_weights_path,
            ModelType.VGG_CIFAR10       : self.get_mlp_weights_path,
            ModelType.VGG_CIFAR100       : self.get_mlp_weights_path
        }         

    def get_mlp_model(self):
        # Implementation of get_resnet_model function
        from model.mnist_model_wgmprop import MNISTModel
        return MNISTModel(input_shape=self.input_shape,
                            nb_class=10)    

    def get_vgg_cifar10_model(self):
        # Implementation of get_resnet_model function
        from model.cifar_models_wgmprop import Cifar10Model
        return Cifar10Model(input_shape=self.input_shape,
                            nb_class=10)  
    
    def get_vgg_cifar100_model(self):
        # Implementation of get_resnet_model function
        from model.cifar_models_wgmprop import Cifar100Model
        return Cifar100Model(input_shape=self.input_shape,
                            nb_class=100)    

    def get_mlp_weights_path(self):
        return self.parameters.model_param.weights_path

    def get_built_model(self):
        try:
            logger.info(f'Loading {self.model_type.value} model')
            model_func        = self.model_funcs[self.model_type]
            weights_path_func = self.weights_path_funcs[self.model_type]
        except KeyError:
            raise ValueError(f"Unsupported model type: {self.parameters.model_type}")
        
        model = model_func()
        if self.model_type == ModelType.MLP:
            model.build(input_shape=(self.parameters.batch_size, self.input_shape[0]*self.input_shape[1]))
        else:
            model.build(input_shape=(self.parameters.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))

        weights_path = weights_path_func()
        try: 
            logger.info(f'Loading weights for {self.model_type.value} model')
            model.load_weights(weights_path, by_name=True) 
        except:
            logger.info(f'WARNING: Cannot load local weights for model {self.parameters.model_type}. Random initialization will be used.')
            pass
                
        return model        

class JsonReader:
    def __init__(self):
        self.param = None

    @staticmethod
    def json2obj(json_config):
        return json.loads(json.dumps(json_config), object_hook=lambda d: SimpleNamespace(**d))

    def read(self, filename):
        assert (filename.endswith(".json"))  # ensures that the config is given as a .json file
        # Opening JSON file
        with open(filename) as json_file:
            config_tmp = json.load(json_file)

        self.param = self.json2obj(config_tmp)
        
class ParamReader:
    def __init__(self, fpath: str):
        self.fpath = fpath
        
    def read(self) -> dict:
        """reads json input file"""
        jsonreader_obj = JsonReader()
        jsonreader_obj.read(filename=self.fpath)
        parameters = jsonreader_obj.param         
        return parameters 

class UserFileProcessor:
    def __init__(self, filename: str):
        self.filename = filename
        self.dir_results_name = 'results'
        self.dir_logs_name    = 'logs'
        
    def read_input_parameters(self) -> dict:
        """reads json input file"""
        jsonreader_obj = JsonReader()
        jsonreader_obj.read(filename=self.filename)
        parameters = jsonreader_obj.param 
        return parameters

    def process_input_parameters(self, parameters: dict) -> Tuple[dict, list, list]:
        """preprocesses the raw parameters to compatible model inputs"""
        # -------------------------------------------------------------------------------------------------
        if parameters.load_weight_at_given_epoch: 
            parameters.chkp_path = parameters.weight_prefix + str(parameters.epoch_to_load_weight) + '.h5'

        parameters.method = PropagationMethod[parameters.method_name]
        intensities = [NoiseIntensity[intensity_.upper()] for intensity_ in parameters.noise_intensities]
        noises = [NoiseType[noisetype_.upper()] for noisetype_ in parameters.noise_types]
        # -------------------------------------------------------------------------------------------------            
        return parameters, noises, intensities

    def create_output_results_directories(self, parameters: dict) -> dict:
        parameters.image_prefix_root = os.path.join(self.dir_results_name, f"{parameters.dataset}", f"{parameters.model_type}", f"{parameters.model_name}")
        os.makedirs(os.path.join(self.dir_results_name, f"{parameters.dataset}"), exist_ok=True)
        os.makedirs(os.path.join(self.dir_results_name, f"{parameters.dataset}", f"{parameters.model_type}"), exist_ok=True)
        os.makedirs(parameters.image_prefix_root, exist_ok=True)
        os.makedirs(os.path.join(self.dir_logs_name, parameters.dataset), exist_ok=True)    
        return parameters
    
    def process(self):
        parameters                      = self.read_input_parameters()
        parameters, noises, intensities = self.process_input_parameters(parameters)
        parameters                      = self.create_output_results_directories(parameters)
        return parameters, noises, intensities
        
        
        