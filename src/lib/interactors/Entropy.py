import numpy as np
from tqdm import tqdm
from .ExperimentalInteractor import ExperimentalInteractor
import matplotlib.pyplot as plt
import os
import scipy.sparse
class Entropy(ExperimentalInteractor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def probabilities_entropy(probabilities):
        return -1*np.sum(probabilities*np.log(probabilities))

    @staticmethod
    def values_entropy(values):
        unique, counts = np.unique(values, return_counts=True)
        values_probability = counts/np.sum(counts)
        return Entropy.probabilities_entropy(values_probability)
    
    @staticmethod
    def get_items_entropy(consumption_matrix):
        lowest_value = np.min(consumption_matrix)
        # mask = np.ones(consumption_matrix.shape[0], dtype=bool)
        # mask[test_uids] = 0
        items_entropy = np.zeros(consumption_matrix.shape[1])
        is_spmatrix = isinstance(consumption_matrix,scipy.sparse.spmatrix)
        if is_spmatrix:
            consumption_matrix = scipy.sparse.csc_matrix(consumption_matrix)
        # consumption_matrix=consumption_matrix.transpose()
        # import time
        for iid in range(consumption_matrix.shape[1]):
            # stime = time.time()
            if is_spmatrix:
                # iid_ratings = consumption_matrix[mask].data
                # iid_ratings = consumption_matrix[mask,:][:,iid].data
                # consumption_matrix.getcol(iid)
                # iid_ratings = consumption_matrix[:,iid][mask].data
                iid_ratings = consumption_matrix[:,iid].A.flatten()
                iid_ratings = iid_ratings[iid_ratings > lowest_value]
            else:
                iid_ratings = consumption_matrix[:,iid]
                iid_ratings = iid_ratings[iid_ratings > lowest_value]
            # print(f"Elapsed time: {time.time()-stime}")
            # unique, counts = np.unique(iid_ratings, return_counts=True)
            # ratings_probability = counts/np.sum(counts)
            # items_entropy[iid] = -1*np.sum(ratings_probability*np.log(ratings_probability))
            items_entropy[iid] = Entropy.values_entropy(iid_ratings)
        return items_entropy

    def train(self,train_dataset):
        super().train(train_dataset)
        self.train_dataset = train_dataset
        self.num_items = self.train_dataset.num_items
        self.unique_values = self.train_dataset.rate_domain
        self.num_unique_values = len(unique_values)
        self.items_ratings = np.zeros((self.num_items,self.num_unique_values))
        self.unique_values_ids = dict(zip(unique_values,list(range(num_unique_values))))
        for uid, iid, reward in train_dataset.data:
            items_ratings[iid,reward] += 1
        # items_entropy = np.power(items_entropy+1,items_entropy+1)
        # fig, ax = plt.subplots()
        # ax.hist(items_entropy,color='k')
        # ax.set_xlabel("Entropy")
        # ax.set_ylabel("#Items")
        # fig.savefig(os.path.join(self.DIRS['img'],"entropy_"+self.get_id()+".png"))
        
    def predict(self,uid,candidate_items,num_req_items):
        items_score =  [self.probabilities_entropy(self.items_ratings[iid]/np.sum(self.items_ratings[iid]))
                        for iid
                        in candidate_items]
        return items_score, None

    def update(self,uid,item,reward,additional_data):
        items_ratings[item,self.unique_values_ids[reward]] += 1 
