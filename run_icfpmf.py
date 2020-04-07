import inquirer
import numpy as np

import interactors
from mf import ICFPMF
from util import DatasetFormatter

dsf = DatasetFormatter()
dsf = dsf.load()
# dsf.get_base()

mf = ICFPMF.getInstance()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
mf.fit(dsf.matrix_users_ratings[dsf.train_uids])

import matplotlib.pyplot as plt

plt.plot(mf.maps)
plt.savefig("img/icfpmf_map.png")
