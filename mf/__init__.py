from .MF import *
from .ICFPMF import *
from .PMF import *
from .SVD import *
from .NMF import *
from .ICFPMFS import *

MF_MODELS = {}
for mf_model_class in [ICFPMFS,SVD,NMF,PMF,ICFPMF]:
    MF_MODELS[mf_model_class.__name__] = mf_model_class
