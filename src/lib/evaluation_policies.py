from collections import defaultdict
from threadpoolctl import threadpool_limits
import scipy.sparse
from lib.utils.Parameterizable import Parameterizable
import numpy as np
import random
from tqdm import tqdm
import interactors


import matplotlib as mpl
import seaborn as sns
from lib.utils.DirectoryDependent import DirectoryDependent
import matplotlib.pyplot as plt
import scipy.stats
import os

class EvaluationPolicy:
    pass

class Interaction(EvaluationPolicy,Parameterizable):
    def __init__(self, num_interactions, interaction_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_interactions = num_interactions
        self.interaction_size = interaction_size
        self.parameters.extend(['num_interactions','interaction_size'])

    def evaluate(self,model,train_dataset,test_dataset):
        with threadpool_limits(limits=1, user_api='blas'):
            test_users = np.unique(test_dataset.data[:,0]).astype(int)
            num_total_items = test_dataset.num_total_items
            test_consumption_matrix = scipy.sparse.csr_matrix((test_dataset.data[:,2],(test_dataset.data[:,0].astype(int),test_dataset.data[:,1].astype(int))),shape=(test_dataset.num_total_users,test_dataset.num_total_items))

            users_items_recommended = defaultdict(list)

            for i in range(len(train_dataset.data)):
                uid = int(train_dataset.data[i,0])
                if uid in test_users:
                    iid = int(train_dataset.data[i,1])
                    reward = train_dataset.data[i,2]
                    users_items_recommended[uid].append(iid)

            num_test_users = len(test_users)
            print(f"Starting {model.__class__.__name__} Training")
            model.train(train_dataset)
            print(f"Ended {model.__class__.__name__} Training")
            users_num_interactions = defaultdict(int)
            available_users = set(test_users)

            history_items_recommended = []

            num_trials = num_test_users*self.num_interactions
            _intervals = num_trials//20
            _num_interactions = 0
            pbar = tqdm(total=num_trials)
            pbar.set_description(f"{model.__class__.__name__}")
            for i in range(num_trials):
                uid = random.sample(available_users,k=1)[0]
                not_recommended = np.ones(num_total_items,dtype=bool)
                not_recommended[users_items_recommended[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]
                items_score, additional_data = model.predict(uid,items_not_recommended,self.interaction_size)
                best_items = items_not_recommended[np.argpartition(items_score,-self.interaction_size)[-self.interaction_size:]]
                users_items_recommended[uid].extend(best_items)

                for item in best_items:
                    history_items_recommended.append((uid,item))
                    model.update(uid,item,test_consumption_matrix[uid,item],additional_data)
                model.increment_time()
                users_num_interactions[uid] += 1
                if users_num_interactions[uid] == self.num_interactions:
                    available_users = available_users - {uid}

                _num_interactions += 1
                if i % _intervals == 0 and i != 0:
                    pbar.update(_num_interactions)
                    _num_interactions = 0

            pbar.update(_num_interactions)
            _num_interactions = 0
            pbar.close()
            return history_items_recommended


class InteractionSample(EvaluationPolicy,Parameterizable):
    def __init__(self, num_interactions, interaction_size, rseed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_interactions = num_interactions
        self.interaction_size = interaction_size
        self.rseed = rseed
        self.parameters.extend(['num_interactions','interaction_size','rseed'])

    def evaluate(self,model,train_dataset,test_dataset):
        np.random.seed(self.rseed)
        test_users = np.unique(test_dataset.data[:,0]).astype(int)
        num_total_items = test_dataset.num_total_items
        test_consumption_matrix = scipy.sparse.csr_matrix((test_dataset.data[:,2],(test_dataset.data[:,0].astype(int),test_dataset.data[:,1].astype(int))),shape=(test_dataset.num_total_users,test_dataset.num_total_items))

        data = np.vstack(
            (train_dataset.data, test_dataset.data))
        from lib.utils.dataset import Dataset

        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()

        consumption_matrix = scipy.sparse.csr_matrix((dataset.data[:,2],(dataset.data[:,0],dataset.data[:,1])),(dataset.num_total_users,dataset.num_total_items))

        num_users_to_sample = 6
        num_consumed = (test_consumption_matrix>0).sum(axis=1).A.flatten()
        users_selected = []
        num_users_to_sample -= len(users_selected)
        uids = np.nonzero(num_consumed >= 100)[0]
        users_sampled = np.random.choice(uids, num_users_to_sample, replace=False)
        users_selected.extend(users_sampled)

        users_items_recommended = defaultdict(list)
        num_users_selected = len(users_selected)
        print(f"Starting {model.__class__.__name__} Training")
        model.train(train_dataset)
        print(f"Ended {model.__class__.__name__} Training")
        users_num_interactions = defaultdict(int)
        available_users = set(users_selected)

        history_items_recommended = []

        num_trials = num_users_selected*self.num_interactions
        _intervals = num_trials//20
        _num_interactions = 0
        pbar = tqdm(total=num_trials)
        pbar.set_description(f"{model.__class__.__name__}")
        for i in range(num_trials):
            uid = random.sample(available_users,k=1)[0]
            not_recommended = np.ones(num_total_items,dtype=bool)
            not_recommended[users_items_recommended[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            items_score, additional_data = model.predict(uid,items_not_recommended,self.interaction_size)
            best_items = items_not_recommended[np.argpartition(items_score,-self.interaction_size)[-self.interaction_size:]]
            users_items_recommended[uid].extend(best_items)

            for item in best_items:
                history_items_recommended.append((uid,item))
                model.update(uid,item,test_consumption_matrix[uid,item],additional_data)
            model.increment_time()
            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.num_interactions:
                available_users = available_users - {uid}

            _num_interactions += 1
            if i % _intervals == 0 and i != 0:
                pbar.update(_num_interactions)
                _num_interactions = 0

        pbar.update(_num_interactions)
        _num_interactions = 0
        pbar.close()
        items_entropy = interactors.Entropy.get_items_entropy(consumption_matrix)
        items_popularity = interactors.MostPopular.get_items_popularity(consumption_matrix,normalize=False)
        for uid, items in users_items_recommended.items():

            colors = mpl.cm.rainbow(np.linspace(0, 1, len(items)))
            fig = plt.figure(figsize=(8,5))
            # plt.colorbar(colors)
            plt.rcParams.update({'font.size': 14})
            plt.scatter(items_entropy, items_popularity, s=100, color='#d1d1d1')

            plt.scatter(items_entropy[items], items_popularity[items], s=100, color=colors)

            plt.text(0, 1.5, 'Correlation: %.4f' % scipy.stats.pearsonr(items_entropy, items_popularity)[0],
                    bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})
            plt.xlabel('Entropy')
            plt.ylabel('Popularity')
            fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],f'plot_{uid}.png'),bbox_inches = 'tight')

        return history_items_recommended

class LimitedInteraction(EvaluationPolicy,Parameterizable):
    def __init__(self, interaction_size, recommend_test_data_rate_limit, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interaction_size = interaction_size
        self.recommend_test_data_rate_limit = recommend_test_data_rate_limit
        self.parameters.extend(['interaction_size','recommend_test_data_rate_limit'])

    def evaluate(self,model,train_dataset,test_dataset):
        with threadpool_limits(limits=1, user_api='blas'):
            test_users = np.unique(test_dataset.data[:,0]).astype(int)
            num_total_items = test_dataset.num_total_items
            test_consumption_matrix = scipy.sparse.csr_matrix((test_dataset.data[:,2],(test_dataset.data[:,0].astype(int),test_dataset.data[:,1].astype(int))),shape=(test_dataset.num_total_users,test_dataset.num_total_items))

            users_items_recommended = defaultdict(list)
            num_test_users = len(test_users)
            print(f"Starting {model.__class__.__name__} Training")
            model.train(train_dataset)
            print(f"Ended {model.__class__.__name__} Training")
            # users_num_interactions = defaultdict(int)
            users_num_items_to_recommend_from_test = dict()
            available_users = set()
            for uid in test_users:
                users_num_items_to_recommend_from_test[uid] = np.floor((test_consumption_matrix[uid] > 0).count_nonzero()*self.recommend_test_data_rate_limit)
                if users_num_items_to_recommend_from_test[uid] > 0:
                    available_users |= {uid}
            
            users_num_items_recommended_from_test = defaultdict(int)

            history_items_recommended = []

            while len(available_users) > 0:
                uid = random.sample(available_users,k=1)[0]
                not_recommended = np.ones(num_total_items,dtype=bool)
                not_recommended[users_items_recommended[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]
                items_score, additional_data = model.predict(uid,items_not_recommended,self.interaction_size)
                best_items = items_not_recommended[np.argpartition(items_score,-self.interaction_size)[-self.interaction_size:]]
                users_items_recommended[uid].extend(best_items)

                for item in best_items:
                    history_items_recommended.append((uid,item))
                    model.update(uid,item,test_consumption_matrix[uid,item],additional_data)
                    users_num_items_recommended_from_test[uid]+=test_consumption_matrix[uid,item]>0

                model.increment_time()
                # users_num_interactions[uid] += 1
                if users_num_items_recommended_from_test[uid] >= users_num_items_to_recommend_from_test[uid]:
                    available_users = available_users - {uid}

            return history_items_recommended
