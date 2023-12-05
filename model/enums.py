from enum import Enum

class DatasetType(Enum):
    CIFAR10  = "cifar10"
    CIFAR100 = "cifar100"
    MNIST    = "mnist"

class ModelType(Enum):
    MLP = "mlp"
    VGG_CIFAR10 = "vgg_cifar10"
    VGG_CIFAR100 = "vgg_cifar100"
    
class PropagationMethod(Enum):
    LPN = "lpn"
    GMM_UT_W_FN = "gmm_ut_w_fn"
    GMM_UT_KL = "gmm_ut_kl"
    MC_UT_FN = "mc_ut_fn"
    MC_UT_LW = "mc_ut_lw"
    MC = "mc"

    @staticmethod
    def get_propagation_type(method):
        if method == PropagationMethod.GMM_UT_W_FN:
            return PropagationType.FN
        elif method == PropagationMethod.GMM_UT_KL:
            return PropagationType.UT
        elif method == PropagationMethod.MC_UT_FN:
            return PropagationType.FN 
        elif method == PropagationMethod.MC_UT_LW:
            return PropagationType.UT 
        elif method == PropagationMethod.LPN:
            return PropagationType.LPN 
        else:
            return None
    
    @staticmethod
    def get_criteria_type(method):
        if method == PropagationMethod.GMM_UT_W_FN:
            return CriteriaType.WASSERSTEIN
        elif method == PropagationMethod.GMM_UT_KL:
            return CriteriaType.KL
        else:
            return None
    
    @staticmethod
    def get_graph_legend(method):
        if method == PropagationMethod.GMM_UT_W_FN:
            return "WGMprop_FN"
        elif method == PropagationMethod.MC_UT_LW:
            return "UT LW"
        elif method == PropagationMethod.MC_UT_FN:
            return "UT FN"
        elif method == PropagationMethod.LPN:
            return "LPN"
        else:
            return None


class PropagationType(Enum):
    HPN = "HPN"
    LPN = "LPN"
    UT = "UT"
    FN = "FN"
    
class CriteriaType(Enum):
    WASSERSTEIN = "wasserstein"
    KL = "KL"

class NoiseIntensity(Enum):
    LIGHT = "light"
    MEDIUM = "medium"
    HIGH = "high"

class NoiseType(Enum):
    BLUR = "blur"
    GAUSSIAN = "gaussian"
    CONTRAST = "contrast"