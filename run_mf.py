import inquirer
import numpy as np

import interactors
import mf
from mf import ICFPMF
from util import DatasetFormatter, GridSearch
import matplotlib.pyplot as plt
import util.metrics as metrics

q = [
    inquirer.Checkbox('mf_models',
                      message='MF models to run',
                      choices=list(mf.MF_MODELS.keys())
                      )
]
answers=inquirer.prompt(q)

dsf = DatasetFormatter()
dsf = dsf.load()

do_gridsearch = True

test_observed_ui = zip(*(dsf.test_consumption_matrix.tocoo().row,dsf.test_consumption_matrix.tocoo().col))
test_ground_truth = dsf.test_consumption_matrix.data


for i in answers['mf_models']:
    model_class=mf.MF_MODELS[i]
    model = model_class(name_prefix=dsf.base)
    # if issubclass(model_class,(mf.ICFPMF,mf.PMF,mf.ICFPMFS)):
    #     model.load_var(dsf.train_consumption_matrix)

    if do_gridsearch:
        if isinstance(model, (mf.PMF,mf.ICFPMFS)):
            parameters = {'var': [0.1,1,10,100,200],
                          'user_var': [0.1,1,10],
                          'item_var': [0.1,1,10]}
        else:
            raise RuntimeError
        gs = GridSearch(model,parameters)
        gs.fit(dsf.train_consumption_matrix)
    else:
        model.fit(dsf.train_consumption_matrix)
        result = model.predict(test_observed_ui)
        print('Test RMSE:',metrics.rmse(result,test_ground_truth))
        model.save()
        if issubclass(model_class,(mf.ICFPMF,mf.ICFPMFS)):
            plt.plot(model.objective_values)
            plt.savefig("img/%s_objective_value.png"%(model_class.__name__))
