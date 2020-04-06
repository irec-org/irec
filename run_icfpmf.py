import inquirer
import numpy as np

import interactors
from mf import ICFPMF
from util import DatasetFormatter

dsf = DatasetFormatter()
# dsf = dsf.load()
dsf.get_base()

mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
mf.fit(dsf.matrix_users_ratings[dsf.train_uids])
