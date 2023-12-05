import os
import numpy as np
from scipy import ndimage
from model.enums import *
import skimage
import tensorflow as tf
import logging
from model.utils_user_inputs import ParamReader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class NoiseGenerator:
    
    def __init__(self, noise_type=NoiseType.BLUR, noise_intensity=NoiseIntensity.MEDIUM, nb_noisy_samples=None):
        self.noise_type         = noise_type
        self.noise_intensity    = noise_intensity
        self.nb_noisy_samples   = nb_noisy_samples
        self.noise_params       = ParamReader(fpath=os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_param.json")).read()
        if self.noise_type.name not in vars(self.noise_params):
            raise ValueError(f"Noise type '{self.noise_type}' not found in the parameters.")
        
        self.selected_params    = vars(vars(self.noise_params)[self.noise_type.name])[self.noise_intensity.name]
        self.noise_get_param_funcs    = {
            NoiseType.BLUR      : self._get_blur_params,
            NoiseType.GAUSSIAN  : self._get_gaussian_params,
            NoiseType.CONTRAST  : self._get_contrast_params
        }
        self.noise_apply_funcs    = {
            NoiseType.BLUR      : self._apply_blur,
            NoiseType.GAUSSIAN  : self._apply_gaussian,
            NoiseType.CONTRAST  : self._apply_contrast
        }                        

    def _get_gaussian_params(self):
        return np.sqrt(self.selected_params.var)

    def _get_blur_params(self):
        return np.abs(np.random.randn(int(self.nb_noisy_samples)) * self.selected_params.sigma_std + self.selected_params.sigma_mean)

    def _get_contrast_params(self):
            alpha = np.random.randn(int(self.nb_noisy_samples)) * self.selected_params.contrast_std + self.selected_params.contrast_mean
            gamma = 0.5 * (1 - alpha)
            return alpha, gamma        

    def get_noise_parameter(self):
        return self.noise_get_param_funcs[self.noise_type]()

    def _apply_blur(self, X_img, noise_parameters):
        return np.array([ndimage.gaussian_filter(X_img, sigma) for sigma in noise_parameters])

    def _apply_gaussian(self, X_img, noise_parameters):
        if tf.is_tensor(X_img):
            X_img = X_img.numpy()
        gimg = np.array([skimage.util.random_noise(X_img, mode="gaussian", mean=0.0, var=noise_parameters**2) for _ in range(int(self.nb_noisy_samples))], dtype=np.float32)
    
        return gimg

    def _apply_contrast(self, X_img, noise_parameters):
        alpha, gamma = noise_parameters
        noisy_imgs = []
        for i in range(len(alpha)):
            contrasted_img = self._add_weighted(X_img, alpha[i], X_img, 0.0, gamma[i])
            contrasted_img = np.clip(contrasted_img, 0.0, 1.0)
            noisy_imgs.append(contrasted_img)
            
        return np.array(noisy_imgs)

    def _add_weighted(self, img1, weight1, img2, weight2, gamma):
        return img1 * weight1 + img2 * weight2 + gamma 

    def concatenate_outputs(self, noisy_images, source_image):
        return np.concatenate([noisy_images, [source_image]], axis=0)

    def generate_noisy_images(self, source_image, noise_parameters):
        return self.noise_apply_funcs[self.noise_type](source_image, noise_parameters)        

    def generate_noise(self, source_image, noise_parameters):
        logger.info(f"call generate_noise")
        return self.concatenate_outputs(noisy_images=self.generate_noisy_images(source_image, noise_parameters), source_image=source_image)

    def generate(self, source_image, cur_label):
        normalized_samples  = self.generate_noise(source_image, noise_parameters=self.get_noise_parameter())
        sample_label        = np.tile(cur_label, (np.shape(normalized_samples)[0],1))
        flatten_images_     = flatten_images(normalized_samples)
        return flatten_images_, sample_label    

def flatten_images(samples):
    shape = samples.shape
    if len(shape) == 3 or len(shape) == 4:
        flatten_images = np.reshape(samples, (shape[0], np.prod(shape[1:])))
    else:
        raise Exception("wrong array")

    return flatten_images
