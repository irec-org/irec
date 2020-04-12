import inquirer
import interactors
from mf import ICFPMF
from util import DatasetFormatter
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

mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
mf = mf.load()
for i in answers['interactors']:

    itr_class = interactors.INTERACTORS[i]
    if issubclass(itr_class,interactors.ICF):
        itr = itr_class.getInstance(var=mf.var,
                                    user_lambda=mf.user_lambda,
                                    consumption_matrix=dsf.matrix_users_ratings
        )
    else:
        itr = itr_class.getInstance(consumption_matrix=dsf.matrix_users_ratings)
        

    if i  == 'LinearThompsonSampling':
        itr.interact(dsf.test_uids, mf.items_means, mf.items_covs)
    elif issubclass(itr_class,interactors.ICF):
        itr.interact(dsf.test_uids, mf.items_means)
    else:
        itr.interact(dsf.test_uids)
        
