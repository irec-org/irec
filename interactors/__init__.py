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
from .ALMostPopular import *
from .ALEntropy import *
from .OurMethod1 import *
from .Entropy0 import *
from .HELF import *
from .PopPlusEnt import *
from .EMostPopular import *
from .DistinctPopular import *

INTERACTORS = {
    # 'Interactor': Interactor,
    # 'ICF': ICF,
    'Distinct Popular': DistinctPopular,
    'ε-Most Popular': EMostPopular,
    'Most Popular': MostPopular,
    'Random': Random,
    'Entropy': Entropy,
    'log(Pop)⋅Ent': LogPopEnt,
    'OurMethod1': OurMethod1,
    'UCB-Learner (MF)': UCBLearner,
    'LinUCB (MF)': LinUCB,
    'Linear ε-Greedy (MF)': LinEGreedy,
    'Linear UCB (PMF)': LinearUCB,
    'GLM-UCB (PMF)': GLM_UCB,
    'TS': ThompsonSampling,
    'Linear ε-Greedy (PMF)': LinearEGreedy,
    'Linear TS (PMF)': LinearThompsonSampling,
    'ε-Greedy': EGreedy,
    'UCB': UCB,
    'Most Representative': MostRepresentative,
    'AL Most Popular': ALMostPopular,
    'AL Entropy': ALEntropy,
    'Entropy0': Entropy0,
    'HELF': HELF,
    'PopPlusEnt': PopPlusEnt,
}
