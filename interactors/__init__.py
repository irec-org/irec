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
from .LinEGreedy import *

INTERACTORS = {
    # 'Interactor': Interactor,
    # 'ICF': ICF,
    'Most Popular': MostPopular,
    'UCBLearner (MF)': UCBLearner,
    'LinUCB (MF)': LinUCB,
    'Linear ε-Greedy (MF)': LinEGreedy,
    'Linear UCB (PMF)': LinearUCB,
    'GLM-UCB (PMF)': GLM_UCB,
    'TS': ThompsonSampling,
    'Linear ε-Greedy (PMF)': LinearEGreedy,
    'Linear TS (PMF)': LinearThompsonSampling,
    'Random': Random,
    'Entropy': Entropy,
    'log(Pop)⋅Ent': LogPopEnt,
    'ε-Greedy': EGreedy,
    'UCB': UCB,
    'Most Representative': MostRepresentative,
}
