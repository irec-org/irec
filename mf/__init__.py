from .MF import *
from .ICFPMF import *
from .PMF import *
from .SVD import *
from .NMF import *

MF_MODELS = {}
for mf_model_class in [ICFPMF,PMF,SVD,NMF]:
    MF_MODELS[mf_model_class.__name__] = mf_model_class
