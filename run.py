import inquirer
import interactors
import mf
import util
from util import DatasetFormatter
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
dsf = DatasetFormatter()
dsf = dsf.load()

util.InteractorsRunner(dsf).run_interactors()
