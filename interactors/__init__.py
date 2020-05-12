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
from .MostRepresentative import *

INTERACTORS = {
    # 'Interactor': Interactor,
    # 'ICF': ICF,
    'LinUCB': LinUCB,
    'Linear ε-Greedy': LinearEGreedy,
    'LinearUCB': LinearUCB,
    'GLM UCB': GLM_UCB,
    'MostPopular': MostPopular,
    'TS': ThompsonSampling,
    'UCBLearner': UCBLearner,
    'LinearTS': LinearThompsonSampling,
    'Random': Random,
    'Entropy': Entropy,
    'log(Pop)*Ent': LogPopEnt,
    'ε-Greedy': EGreedy,
    'UCB': UCB,
    'MostRepresentative': MostRepresentative,
}
