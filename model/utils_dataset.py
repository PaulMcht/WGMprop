import os
import numpy as np
import tensorflow as tf
import pickle
import logging
import itertools
from model.enums import ModelType
from model.enums import DatasetType

from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class InputPreprocessor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_type = ModelType(model_name.lower())
        self.prepocessor_funcs = {
            ModelType.MLP : self._preprocess_mlp,
            ModelType.VGG_CIFAR10 : self._preprocess_vgg_cifar10,
            ModelType.VGG_CIFAR100 : self._preprocess_vgg_cifar100
        }

    def _preprocess_mlp(self, x): 
        # Implementation of preprocess_mlp function
        return x / 255.0
    
    def _preprocess_vgg_cifar10(self, x): 
        # Implementation of preprocess_vgg_cifar10 function
        return x/255.0
    
    def _preprocess_vgg_cifar100(self, x): 
        # Implementation of preprocess_vgg_cifar100 function
        return x/255.0
        
    def _convert_labels_to_one_hot(self, y_test): 
        return tf.where(tf.equal(tf.one_hot(y_test, 10), 0), 0.1, 0.9)   

    def preprocess(self, x_test, y_test):
        try:
            logger.info(f'Preprocessing inputs for {self.model_type.value} model')
            model_func = self.prepocessor_funcs[self.model_type]
        except KeyError:
            raise ValueError(f"Unsupported preprocessing for model type: {self.model_name}")

        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
        x_test = model_func(x_test)
        y_test = self._convert_labels_to_one_hot(y_test)
        del model_func
        return x_test, y_test


class DatasetLoader:
    def __init__(self, dataset_name, cache_dir="./cache"):
        self.dataset_name = dataset_name
        self.cache_dir    = cache_dir
        self.dataset_type = DatasetType(dataset_name.lower())
        self.loader_funcs = {
            DatasetType.CIFAR10:  self.load_cifar10_dataset,
            DatasetType.CIFAR100: self.load_cifar100_dataset,
            DatasetType.MNIST:    self.load_mnist_dataset
        }
        self.loader_params = {
            DatasetType.CIFAR10:  {"filename": "cifar-10-batches-py.tar.gz",  "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"},
            DatasetType.CIFAR100: {"filename": "cifar-100-batches-py.tar.gz", "url": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"},
            DatasetType.MNIST:    {"filename": "mnist-batches-py.tar.gz",     "url": "https://www.cs.toronto.edu/~kriz/mnist-python.tar.gz"}
        }  
        self.loader_funcs_local = {
            DatasetType.CIFAR10:  self.load_dataset_from_local_directory,
            DatasetType.CIFAR100: self.load_dataset_from_local_directory,
            DatasetType.MNIST:    self.load_dataset_from_local_directory
        }        
        self.loader_params_local = {
            DatasetType.CIFAR10:  "cifar-10-batches-py",
            DatasetType.CIFAR100: "cifar-100-batches-py",
            DatasetType.MNIST:    "mnist-batches-py"
        }          

    def load_mnist_dataset(self, filename, url):
        # Implementation of load_cifar10_dataset function
        if os.path.exists(os.path.join(self.cache_dir, filename)):   
            cache_subdir = tf.keras.utils.get_file(fname=filename, origin=url, untar=True)                
            logging.info(f"Loading MNIST dataset from local copy...")
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(cache_subdir=cache_subdir)
            del cache_subdir
        else:
            logging.info(f"Downloading MNIST dataset...")
            try:
                (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
            except: 
                logging.info(f"Downloading MNIST dataset failed...")
                logging.info(f"Loading local dataset instead in {'/datasets/mnist/mnist-batches-py.tar.gz'}...")
                cache_subdir               = tf.keras.utils.get_file(self.loader_params_local[DatasetType.MNIST], origin=os.path.join("file://"+ os.getcwd(), "datasets", "mnist", "mnist-batches-py.tar.gz"), untar=True, extract=True)
                test_images, test_labels   = self.loader_funcs_local[DatasetType.MNIST](folder_name=self.loader_params_local[DatasetType.MNIST])
                train_images, train_labels = None, None
                del cache_subdir


        return train_images, train_labels, test_images, test_labels
    
    def load_dataset_from_local_directory(self, folder_name):
        cur_dict    = unpickle(os.path.join(os.getcwd(), "data", folder_name, "test_batch"))                
        test_images = np.array(cur_dict[b'data'])
        test_labels = np.array(cur_dict[b'labels'], dtype=np.int32)        
        return test_images, test_labels

    def load_cifar10_dataset(self, filename, url):
        # Implementation of load_cifar10_dataset function
        if os.path.exists(os.path.join(self.cache_dir, filename)):   
            cache_subdir = tf.keras.utils.get_file(fname=filename, origin=url, untar=True)                
            logging.info(f"Loading CIFAR-10 dataset from local copy...")
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data(cache_subdir=cache_subdir)
            del cache_subdir
        else:
            logging.info(f"Downloading CIFAR-10 dataset...")
            try:
                (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
            except: 
                logging.info(f"Downloading CIFAR-10 dataset failed...")
                logging.info(f"Loading local dataset instead in {'/datasets/cifar10/cifar-10-python.tar.gz'}...")
                cache_subdir               = tf.keras.utils.get_file(self.loader_params_local[DatasetType.CIFAR10], origin=os.path.join("file://"+ os.getcwd(), "datasets", "cifar10", "cifar-10-python.tar.gz"), untar=True, extract=True)
                test_images, test_labels   = self.loader_funcs_local[DatasetType.CIFAR10](folder_name=self.loader_params_local[DatasetType.CIFAR10])
                train_images, train_labels = None, None
                del cache_subdir


        return train_images, train_labels, test_images, test_labels

    def load_cifar100_dataset(self, filename, url):
        # Implementation of load_cifar100_dataset function
        if os.path.exists(os.path.join(self.cache_dir, filename)):   
            logging.info(f"Loading CIFAR-100 dataset from local copy...")
            cache_subdir = tf.keras.utils.get_file(fname=filename, origin=url, untar=True)           
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data(cache_subdir=cache_subdir)
            del cache_subdir
        else:
            logging.info(f"Downloading CIFAR-100 dataset...")
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

        return train_images, train_labels, test_images, test_labels
    
    def load_mnist_dataset(self, filename, url):
        # Implementation of load_mnist_dataset function
        if os.path.exists(os.path.join(self.cache_dir, filename)): 
            logging.info(f"Loading MNIST dataset from local copy...")
            cache_subdir = tf.keras.utils.get_file(fname=filename, origin=url, untar=True)              
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(cache_subdir=cache_subdir)
            del cache_subdir
        else:
            logging.info(f"Downloading MNIST dataset...")
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        return train_images, train_labels, test_images, test_labels
    
    def load_test_dataset(self):
        try:
            logger.info(f'Loading dataset {self.dataset_name}')
            loader_func     = self.loader_funcs[self.dataset_type]
            loader_param    = self.loader_params[self.dataset_type]             
        except KeyError:
            raise ValueError(f"Cannot load dataset {self.dataset_name}. Available datasets are CIFAR10, CIFAR100 and MNIST")

        _, _, x_test, y_test = loader_func(**loader_param)
        del loader_func, loader_param
        return x_test, y_test


def get_input_shape(x_test):
    return tf.shape(x_test[0])

def get_test_cases(intensities, noises, nb_test, max=100):
    return [(intensity, noise, i) for intensity, noise in itertools.product(intensities, noises) for i in np.random.randint(0,max,nb_test)]  

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict