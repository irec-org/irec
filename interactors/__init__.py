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
from .PPELPE import *
from .kNNBandit import *
from .COFIBA import *
from .OurMethod2 import *

INTERACTORS = {
    'COFIBA': COFIBA,
    'TS': ThompsonSampling,
    'Random': Random,
    'kNN-Bandit': kNNBandit,
    'AL Most Popular': ALMostPopular,
    'OurMethod1': OurMethod1,
    'OurMethod2': OurMethod2,
    'Most Popular': MostPopular,
    'Linear UCB (PMF)': LinearUCB,
    'Linear ε-Greedy (MF)': LinEGreedy,
    'LinUCB (MF)': LinUCB,
    'PPELPE': PPELPE,
    'UCB-Learner (MF)': UCBLearner,
    'GLM-UCB (PMF)': GLM_UCB,
    'Linear ε-Greedy (PMF)': LinearEGreedy,
    'Linear TS (PMF)': LinearThompsonSampling,
    'ε-Greedy': EGreedy,
    'UCB': UCB,
    'Most Representative': MostRepresentative,
    'AL Entropy': ALEntropy,
    'Entropy0': Entropy0,
    'HELF': HELF,
    'Pop+Ent': PopPlusEnt,
    'Distinct Popular': DistinctPopular,
    'ε-Most Popular': EMostPopular,
    'Entropy': Entropy,
    'log(Pop)⋅Ent': LogPopEnt,
}
