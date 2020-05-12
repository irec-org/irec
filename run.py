import inquirer
import interactors
import mf
from util import DatasetFormatter
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
q = [
    inquirer.Checkbox('interactors',
                      message='Interactors to run',
                      choices=list(interactors.INTERACTORS.keys())
                      )
]
answers=inquirer.prompt(q)
interactors_classes = list(map(lambda x:interactors.INTERACTORS[x],answers['interactors']))

dsf = DatasetFormatter()
dsf = dsf.load()

if np.any(list(map(
        lambda itr_class: issubclass(itr_class,interactors.ICF),
        interactors_classes))):
    mf_model = mf.ICFPMF()
    mf_model.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    mf_model = mf_model.load()

if np.any(list(map(
        lambda itr_class: issubclass(itr_class,(
            interactors.LinUCB,
            interactors.MostRepresentative)),
        interactors_classes
        ))):
    svd_model = mf.SVD()
    svd_model = svd_model.load()
    u, s, vt = scipy.sparse.linalg.svds(
        scipy.sparse.csr_matrix(dsf.matrix_users_ratings[dsf.train_uids]),
        k=10)
    Q = svd_model.items_weights

for itr_class in interactors_classes:

    if issubclass(itr_class,interactors.ICF):
        itr = itr_class(var=mf_model.var,
                        user_lambda=mf_model.user_lambda,
                        consumption_matrix=dsf.matrix_users_ratings,
                        prefix_name=dsf.base
        )
    else:
        itr = itr_class(consumption_matrix=dsf.matrix_users_ratings,
                        prefix_name=dsf.base)
        
    if issubclass(itr_class,interactors.LinearThompsonSampling):
        itr.interact(dsf.test_uids, mf_model.items_means, mf_model.items_covs)
    elif issubclass(itr_class,interactors.ICF):
        itr.interact(dsf.test_uids, mf_model.items_means)
    elif issubclass(itr_class,(interactors.LinUCB)):
        itr.interact(dsf.test_uids,Q)
    elif issubclass(itr_class,interactors.UCBLearner):
        itr.interact(dsf.test_uids,Q)
    elif issubclass(itr_class,interactors.MostRepresentative):
        itr.interact(dsf.test_uids,Q)
    else:
        itr.interact(dsf.test_uids)
