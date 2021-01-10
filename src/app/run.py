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

ir = util.InteractorsRunner(dsf)
ir.select_interactors()
ir.run_interactors()
# ir.run_bases(['tr_te_yahoo_music',
#               'tr_te_good_books','tr_te_ml_10m'])
