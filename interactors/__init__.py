import random
import numpy as np

from .Interactor import *
from .ICF import *
from .LinearUCB import *
from .ThompsonSampling import *
from .LinearEGreedy import *
from .GLM_UCB import *


INTERACTORS = {
    'Interactor': Interactor,
    'ICF': ICF,
    'LinearUCB': LinearUCB,
    'ThompsonSampling': ThompsonSampling,
    'LinearEGreedy': LinearEGreedy,
    'GLM_UCB': GLM_UCB,
}
