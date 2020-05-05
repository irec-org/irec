import inquirer
import interactors
from mf import ICFPMF
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
    mf = ICFPMF()
    mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
    mf = mf.load()
for i in answers['interactors']:

    itr_class = interactors.INTERACTORS[i]
    if issubclass(itr_class,interactors.ICF):
        itr = itr_class(var=mf.var,
                                    user_lambda=mf.user_lambda,
                                    consumption_matrix=dsf.matrix_users_ratings
        )
    else:
        itr = itr_class(consumption_matrix=dsf.matrix_users_ratings)
        

    if i  == 'LinearThompsonSampling':
        itr.interact(dsf.test_uids, mf.items_means, mf.items_covs)
    elif issubclass(itr_class,interactors.ICF):
        itr.interact(dsf.test_uids, mf.items_means)
    elif issubclass(itr_class,interactors.LinUCB) or issubclass(itr_class,interactors.UCBLearner):
        # model = NMF(n_components=10, init='nndsvd', random_state=0)
        # P = model.fit_transform(dsf.matrix_users_ratings[dsf.train_uids])
        # Q = model.components_.T
        # itr.interact(dsf.test_uids,Q)
        u, s, vt = scipy.sparse.linalg.svds(
            scipy.sparse.csr_matrix(dsf.matrix_users_ratings[dsf.train_uids]),
            k=10)
        itr.interact(dsf.test_uids,vt.T)
    else:
        itr.interact(dsf.test_uids)
        
