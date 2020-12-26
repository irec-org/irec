import inquirer
import interactors
import mf
from util import DatasetFormatter
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.sparse
import os

q = [
    inquirer.List('mf_model',
                      message='MF model to run',
                      choices=list(mf.MF_MODELS.keys())
                      )
]
answers=inquirer.prompt(q)
model_name = answers['mf_model']
dsf = DatasetFormatter()
dsf = dsf.load()
model_class=mf.MF_MODELS[model_name]
model = model_class(name_prefix=dsf.base)
correlations= []
items_popularity = interactors.MostPopular.get_items_popularity(dsf.train_consumption_matrix)
latent_factors_numbers = list(range(1,500,40))
for k in latent_factors_numbers:
    model.num_lat = k
    model.fit(dsf.train_consumption_matrix)

    items_representativeness = interactors.MostRepresentative.get_items_representativeness(model.items_weights)
    rep_mp_corr = np.corrcoef(items_popularity,items_representativeness)[0,1]
    correlations.append(rep_mp_corr)


fig, ax = plt.subplots()
ax.set_ylabel("Correlation to Most Popular")
ax.set_xlabel("Number of Latent Factors")
ax.set_title(f"{dsf.base} {model_name}")
ax.plot(latent_factors_numbers,correlations)

fig.savefig(os.path.join(dsf.DIRS['img'],f"{dsf.PRETTY[dsf.base]}_{model_name}_mp_corr.png"))


