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
# dsf = dsf.load()
dsf.get_base()

mf = ICFPMF()
mf.load_var(dsf.matrix_users_ratings[dsf.train_uids])
mf = mf.load()
for i in answers['interactors']:

    if i == 'MostPopular':
        itr = interactors.INTERACTORS[i](consumption_matrix=dsf.matrix_users_ratings)
    else:
        itr = interactors.INTERACTORS[i](var=mf.var,
                                        user_lambda=mf.user_lambda,
                                        consumption_matrix=dsf.matrix_users_ratings
        )

    if i  == 'ThompsonSampling':
        itr.interact(dsf.test_uids, mf.items_means, mf.items_covs)
    elif i == 'MostPopular':
        itr.interact(dsf.test_uids)
    else:
        itr.interact(dsf.test_uids, mf.items_means)
    
