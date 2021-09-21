import numpy as np
from tqdm import tqdm
from . import ValueFunction, Entropy, Entropy0, MostPopular, LogPopEnt, ExperimentalValueFunction
import matplotlib.pyplot as plt
import scipy.stats
import os
import random
import metrics


class DistinctPopular(ExperimentalValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.num_total_items = self.train_dataset.num_total_items

        self.items_entropy = Entropy.get_items_entropy(
            self.train_consumption_matrix)
        np.seterr('warn')
        self.items_popularity = MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)
        self.items_distance = metrics.get_items_distance(
            self.train_consumption_matrix)

        self.top_iids = defaultdict([])
        # num_total_items = self.train_consumption_matrix.shape[1]

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        if len(self.top_iids) > 0:
            candidate_items_distance = np.mean(
                self.items_distance[self.top_iids[uid]][:, candidate_items],
                axis=0)
        else:
            candidate_items_distance = 1

        return candidate_items_distance * items_popularity[cadidate_items], None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        self.top_iids[uid].append(item)

        # correlation = scipy.stats.pearsonr(items_entropy,items_popularity)[0]

        # fig, ax = plt.subplots()
        # ax.scatter(items_entropy,items_popularity,marker="D",color='darkblue')
        # ax.set_ylabel("Popularity")
        # ax.set_xlabel("Entropy")
        # ax.text(0.3, 0.9 , f'Correlation coefficient: {correlation:.2f}', color='k',
        #         ha='center', va='center',
        #         bbox=dict(facecolor='none', edgecolor='k', pad=10.0),
        #         transform = ax.transAxes)
        # for start, end, color in [(0,10,'green'),(10,20,'red'),(20,30,'darkred'),(30,40,'yellow'),(40,50,'orange')]:
        #     ax.scatter(items_entropy[top_iids[start:end]],items_popularity[top_iids[start:end]],marker='D',color=color)
        #     print("[%d,%d] sum(popularity)=%.2f sum(entropy)=%.2f"%(start,end,
        #                                                          np.sum(items_popularity[top_iids[start:end]]/np.max(items_popularity)),
        #                                                          np.sum(items_entropy[top_iids[start:end]]/np.max(items_entropy))))
        # fig.savefig(os.path.join(self.DIRS['img'],"corr_popent_"+self.get_id()+".png"))

        # num_total_users = len(uids)
        # for idx_uid in tqdm(range(num_total_users)):
        #     uid = uids[idx_uid]
        #     self.results[uid].extend(top_iids)
        # self.save_results()
