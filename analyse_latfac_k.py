import inquirer
import interactors
import mf
from util import DatasetFormatter
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import scipy.sparse

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
for k in range(1,3,1):
    model.num_lat = k
    model.fit(dsf.train_consumption_matrix)

    items_representativeness = interactors.MostRepresentative.get_items_representativeness(model.items_weights)
    rep_weights_corr = np.corrcoef(items_representativeness,np.norm(model.items_weights,axis=0))[0,1]
    correlations.append(rep_weights_corr)


fig, ax = plt.subplots()
ax.set_ylabel("Correlation to Most Popular")
ax.set_xlabel("Number of Latent Factors")
ax.set_title(f"{dsf.base} {model_name}")
ax.plot(correlations)

fig.savefig(os.path.join(self.DIRS['img'],f"{dsf.base}_{model_name}_mp_corr.png"))


