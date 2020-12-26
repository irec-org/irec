import inquirer
import interactors
import mf
import util
from util import DatasetFormatter, MetricsEvaluator
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
import recommenders
dsf = DatasetFormatter()
dsf = dsf.load()
THRESHOLD = interactors.Interactor().threshold

ground_truth = MetricsEvaluator.get_ground_truth(dsf.consumption_matrix,THRESHOLD)

pmf_model = mf.ICFPMFS(name_prefix=dsf.base)
pmf_model.load_var(dsf.train_consumption_matrix)

history_rates_to_train = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

test_matrix = dsf.test_consumption_matrix
recommenders_class = [recommenders.UserKNN]

for history_rate in history_rates_to_train:
    print('%.2f%% of history'%(history_rate*100))
    for interactor_class in [
            # interactors.Entropy,
            interactors.MostPopular,
            # interactors.Random,
            # interactors.LogPopEnt,
            interactors.LinearUCB,
            interactors.LinEGreedy,
            interactors.OurMethod1,
    ]:
        if issubclass(interactor_class,interactors.ICF):
            interactor_model = interactor_class(
                name_prefix=dsf.base,
                var=pmf_model.var,
                user_lambda=pmf_model.get_user_lambda(),
            )
        else:
            interactor_model = interactor_class(
                name_prefix=dsf.base)
                # interactions=100,
                # interaction_size=5)
        interactor_model.results = interactor_model.load_results()
        train_matrix = dsf.train_consumption_matrix.tolil()
        users_rel_items_history = dict()

        for uid, items in interactor_model.results.items():
            items = np.array(items)
            users_rel_items_history[uid] = []
            user_history_size = 0
            aux = np.isin(items,ground_truth[uid])
            user_num_consumed_items = np.count_nonzero(aux)
            for item in items[aux]:
                if user_history_size <= history_rate*user_num_consumed_items:
                    users_rel_items_history[uid].append((item,dsf.consumption_matrix[uid,item]))
                    train_matrix[uid,item] = dsf.consumption_matrix[uid,item]
                    user_history_size += 1
        train_matrix = train_matrix.tocsr()

        print('\t*',interactor_class.__name__)
        for recommender_class in recommenders_class:
            print('\t\t-',recommender_class.__name__)
            recommender_model = recommender_class(name_prefix=dsf.base,
                                                  name_suffix=interactor_model.get_name()+'_history_rate_%.2f'%(history_rate))
            recommender_model.train(train_matrix)
            recommender_model.predict(test_matrix)
