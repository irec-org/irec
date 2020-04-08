import random
import numpy as np

from .Interactor import *
from .ICF import *
from .LinearUCB import *
from .ThompsonSampling import *
from .LinearEGreedy import *
from .GLM_UCB import *
from .MostPopular import *
from .Random import *


INTERACTORS = {
    # 'Interactor': Interactor,
    # 'ICF': ICF,
    'LinearEGreedy': LinearEGreedy,
    'LinearUCB': LinearUCB,
    'GLM_UCB': GLM_UCB,
    'ThompsonSampling': ThompsonSampling,
    'MostPopular': MostPopular,
    'Random': Random,
}
