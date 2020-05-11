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

dsf = DatasetFormatter()
dsf = dsf.load()
# dsf.get_base()
if np.any([issubclass(interactors.INTERACTORS[i],interactors.ICF) for i in answers['interactors']]):
    mf_model = mf.ICFPMF()
    mf_model.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    mf_model = mf_model.load()


if np.any([issubclass(interactors.INTERACTORS[i],interactors.LinUCB) or
           issubclass(interactors.INTERACTORS[i],interactors.UCBLearner) or
           issubclass(interactors.INTERACTORS[i],interactors.MostRepresentative)
    for i in answers['interactors']]):
    u, s, vt = scipy.sparse.linalg.svds(
        scipy.sparse.csr_matrix(dsf.matrix_users_ratings[dsf.train_uids]),
        k=10)
    Q = vt.T
    # model = NMF(n_components=10, init='nndsvd', random_state=0)
    # P = model.fit_transform(dsf.matrix_users_ratings[dsf.train_uids])
    # Q = model.components_.T

for i in answers['interactors']:

    itr_class = interactors.INTERACTORS[i]
    if issubclass(itr_class,interactors.ICF):
        itr = itr_class(var=mf_model.var,
                        user_lambda=mf_model.user_lambda,
                        consumption_matrix=dsf.matrix_users_ratings,
                        prefix_name=dsf.base
        )
    else:
        itr = itr_class(consumption_matrix=dsf.matrix_users_ratings,
                        prefix_name=dsf.base)
        
    if i  == 'LinearThompsonSampling':
        itr.interact(dsf.test_uids, mf_model.items_means, mf_model.items_covs)
    elif issubclass(itr_class,interactors.ICF):
        itr.interact(dsf.test_uids, mf_model.items_means)
    elif issubclass(itr_class,interactors.LinUCB):
        itr.interact(dsf.test_uids,Q)
    elif issubclass(itr_class,interactors.UCBLearner):
        itr.interact(dsf.test_uids,Q)
    elif issubclass(itr_class,interactors.MostRepresentative):
        itr.interact(dsf.test_uids,Q)
    else:
        itr.interact(dsf.test_uids)
