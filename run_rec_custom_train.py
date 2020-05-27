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

history_rates_to_train = [0.2,0.5,0.7,0.9]

recommenders_class = [recommenders.SVDPlusPlus]

for history_rate in history_rates_to_train:
    print('%.2f%% of history'%(history_rate*100))
    for interactor_class in [interactors.Entropy,
                            interactors.MostPopular,
                            interactors.Random]:
        interactor_model = interactor_class(
            name_prefix=dsf.base,
            interations=dsf.num_items,
            interaction_size=1)
        interactor_model.results = interactor_model.load_results()
        train_matrix = dsf.consumption_matrix.tolil()
        test_matrix = scipy.sparse.lil_matrix(train_matrix.shape)
        users_rel_items_history = dict()

        # if np.any(list(map(lambda x: x in [recommenders.SVDPlusPlus],recommenders_class))):
        #     svdplusplus_mf_model = mf.SVDPlusPlus(name_prefix=dsf.base)
        #     svdplusplus_mf_model()

        for uid, items in interactor_model.results.items():
            users_rel_items_history[uid] = []
            user_history_size = 0
            user_num_consumed_items = len(ground_truth[uid])
            for item in items:
                if item in ground_truth[uid]:
                    if user_history_size > history_rate*user_num_consumed_items:
                        test_matrix[uid,item] = train_matrix[uid,item]
                        train_matrix[uid,item] = 0
                    else:
                        users_rel_items_history[uid].append((item,dsf.consumption_matrix[uid,item]))
                        user_history_size += 1
        test_matrix = test_matrix.tocsr()
        # print(test_matrix.__repr__)
        train_matrix = train_matrix.tocsr()
        # print(train_matrix.__repr__)

        # ground_truth = MetricsEvaluator.get_ground_truth(test_matrix,THRESHOLD)

        # items_popularity = interactors.MostPopular.get_items_popularity(dsf.consumption_matrix,[],normalize=True)

        print('\t*',interactor_class.__name__)
        for recommender_class in recommenders_class:
            print('\t\t-',recommender_class.__name__)
            recommender_model = recommender_class(name_prefix=dsf.base,
                                                  name_suffix=interactor_class.__name__+'_history_rate_%.2f'%(history_rate))
            recommender_model.train(train_matrix)
            recommender_model.predict(test_matrix)
            # me = MetricsEvaluator(name=recommender_model.get_name(),k=recommender_model.result_list_size,threshold=THRESHOLD)
            # me.eval_metrics(recommender_model.results, )

            
