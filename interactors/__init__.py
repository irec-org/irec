import random
import numpy as np

from .Interactor import *
from .ICF import *
from .LinearUCB import *
from .LinearThompsonSampling import *
from .LinearEGreedy import *
from .GLM_UCB import *
from .MostPopular import *
from .Random import *
from .Entropy import *
from .LogPopEnt import *
from .ThompsonSampling import *
from .EGreedy import *
from .UCB import *
from .LinUCB import *
from .UCBLearner import *

INTERACTORS = {
    # 'Interactor': Interactor,
    # 'ICF': ICF,
    'LinearEGreedy': LinearEGreedy,
    'LinearUCB': LinearUCB,
    'MostPopular': MostPopular,
    'ThompsonSampling': ThompsonSampling,
    'LinUCB': LinUCB,
    'UCBLearner': UCBLearner,
    'GLM_UCB': GLM_UCB,
    'LinearThompsonSampling': LinearThompsonSampling,
    'Random': Random,
    'Entropy': Entropy,
    'LogPopEnt': LogPopEnt,
    'EGreedy': EGreedy,
    'UCB': UCB,
}
