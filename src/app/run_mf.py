import inquirer
import numpy as np

import value_functions
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

do_gridsearch = False

test_observed_ui = (dsf.test_consumption_matrix.tocoo().row,dsf.test_consumption_matrix.tocoo().col)
test_ground_truth = dsf.test_consumption_matrix.data


train_observed_ui = (dsf.train_consumption_matrix.tocoo().row,dsf.train_consumption_matrix.tocoo().col)
train_ground_truth = dsf.train_consumption_matrix.data


for i in answers['mf_models']:
    model_class=mf.MF_MODELS[i]
    model = model_class(name_prefix=dsf.base)
    if issubclass(model_class,(mf.ICFPMF,mf.PMF,mf.ICFPMFS)):
        model.load_var(dsf.train_consumption_matrix)

    if do_gridsearch:
        if isinstance(model, (mf.PMF,mf.ICFPMFS)):
            parameters = {
                'var': np.linspace(0.1,30,10),
                'user_var': [1],
                'item_var': [1],
            }
        else:
            raise RuntimeError
        gs = GridSearch(model,parameters)
        gs.fit(dsf.train_consumption_matrix)
    else:
        model.fit(dsf.train_consumption_matrix)
        result = model.predict(train_observed_ui)
        print('Train RMSE:',metrics.rmse(result,train_ground_truth))
        result = model.predict(test_observed_ui)
        print('Test RMSE:',metrics.rmse(result,test_ground_truth))
        model.save()
        if issubclass(model_class,(mf.ICFPMF,mf.ICFPMFS)):
            plt.plot(model.objective_values[2:])
            plt.savefig("img/%s_objective_value.png"%(model_class.__name__))
