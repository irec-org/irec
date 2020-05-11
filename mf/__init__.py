from .MF import *
from .ICFPMF import *
from .PMF import *

MF_MODELS = {}
for mf_model_class in [ICFPMF,PMF]:
    MF_MODELS[mf_model_class.__name__] = mf_model_class
