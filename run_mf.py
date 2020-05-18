import inquirer
import numpy as np

import interactors
import mf
from mf import ICFPMF
from util import DatasetFormatter
import matplotlib.pyplot as plt

q = [
    inquirer.Checkbox('mf_models',
                      message='MF models to run',
                      choices=list(mf.MF_MODELS.keys())
                      )
]
answers=inquirer.prompt(q)

dsf = DatasetFormatter()
dsf = dsf.load()
for i in answers['mf_models']:
    model_class=mf.MF_MODELS[i]
    model = model_class(name_prefix=dsf.base)
    if issubclass(model_class,(mf.ICFPMF,mf.PMF,mf.ICFPMFS)):
        model.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    model.fit(dsf.matrix_users_ratings[dsf.train_uids])
    if issubclass(model_class,(mf.ICFPMF,mf.ICFPMFS)):
        plt.plot(model.objective_values)
        plt.savefig("img/%s_objective_value.png"%(model_class.__name__))
