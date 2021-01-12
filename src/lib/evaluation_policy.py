from collections import defaultdict
import scipy.sparse
from utils.Parameterizable import Parameterizable
import numpy as np
import random

class EvaluationPolicy:
    pass

class Interaction(EvaluationPolicy,Parameterizable):
    def __init__(self, num_interactions, interaction_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_interactions = num_interactions
        self.interaction_size = interaction_size
        self.parameters.extend(['num_interactions','interaction_size'])

    def evaluate(self,model,train_dataset,test_dataset):
        test_users = np.unique(test_dataset.data[:,0])
        # total_num_users = len(np.unique(np.concatenate((train_data[0],
        #                                             test_data[0]),axis=None)))

        # total_num_items = len(np.unique(np.concatenate((train_data[1],
        #                                             test_data[1]),axis=None)))
        
        num_items = test_dataset.num_items
        # print(np.max(test_data[:,0]),test_dataset.num_items)
        # print(np.max(test_data[:,1]),test_dataset.num_users)
        # print(test_data)
        test_consumption_matrix = scipy.sparse.csr_matrix((test_dataset.data[:,2],(test_dataset.data[:,0].astype(int),test_dataset.data[:,1].astype(int))),shape=(test_dataset.num_users,test_dataset.num_items))
        users_items_recommended = defaultdict(list)
        num_test_users = len(test_users)
        model.train(train_dataset.data)
        users_num_interactions = defaultdict(int)
        available_users = set(test_users)
        for i in range(num_test_users*self.num_interactions):
            uid = random.sample(available_users,k=1)[0]
            print(uid)
            # for i in range(self.interaction_size):
            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[users_items_recommended[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            items_score, additional_data = model.predict(uid,items_not_recommended,self.interaction_size)
            best_items = list(reversed(np.argsort(items_score)))[:self.interaction_size]
            # best_item = items_not_recommended[np.argmax(items_score)]
            users_items_recommended[uid].extend(best_items)

            # for i in range(self.interaction_size):
            # for item in users_items_recommended[users_num_interactions[uid]*self.interaction_size:(users_num_interactions[uid]+1)*self.interaction_size]:
            for item in best_items:
                model.update(uid,item,test_consumption_matrix[uid,item],additional_data)
            # model.increment_time()
            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.num_interactions:
                available_users = available_users - {uid}
