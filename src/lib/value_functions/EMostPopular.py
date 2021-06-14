import numpy as np
from tqdm import tqdm
from . import ValueFunction, Entropy, Entropy0, MostPopular, LogPopEnt, ExperimentalValueFunction
import matplotlib.pyplot as plt
import scipy.stats
import os
import random


class EMostPopular(ExperimentalValueFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def top_emostpopular(self, items_exploitation, items_exploration):
        top_iids = []
        num_total_items = self.train_consumption_matrix.shape[1]
        for i in range(num_total_items):
            not_recommended = np.ones(num_total_items, dtype=bool)
            not_recommended[top_iids] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            if self.epsilon < np.random.rand():
                best_item = items_not_recommended[np.argmax(
                    items_exploitation[items_not_recommended])]
            else:
                # best_item = random.choices(items_not_recommended,
                #                            weights=items_entropy[items_not_recommended]
                #                            ,k=1)[0]
                best_item = items_not_recommended[np.argmax(
                    items_exploration[items_not_recommended])]
            top_iids.append(best_item)
        return top_iids

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)
        self.train_dataset = train_dataset
        self.train_consumption_matrix = scipy.sparse.csr_matrix(
            (self.train_dataset.data[:, 2],
             (self.train_dataset.data[:, 0], self.train_dataset.data[:, 1])),
            (self.train_dataset.num_total_users,
             self.train_dataset.num_total_items))
        self.items_entropy = Entropy.get_items_entropy(
            self.train_consumption_matrix)
        self.items_popularity = MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)

        self.top_iids = self.top_emostpopular(self.items_popularity,
                                              self.items_entropy)
        self.items_score = np.empty(len(self.top_iids))
        for i, iid in enumerate(reversed(self.top_iids)):
            self.items_score[iid] = i

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_score[candidate_items]
        return items_score, None

    def update(self, observation, action, reward, info):
        uid = action[0]
        item = action[1]
        additional_data = info
        pass
        # top_iids = []
        # num_total_items = self.train_consumption_matrix.shape[1]
        # for i in range(num_total_items):
        #     not_recommended = np.ones(num_total_items,dtype=bool)
        #     not_recommended[top_iids] = 0
        #     items_not_recommended = np.nonzero(not_recommended)[0]
        #     if self.epsilon < np.random.rand():
        #         best_item = items_not_recommended[np.argmax(items_popularity[items_not_recommended])]
        #     else:
        #         # best_item = random.choices(items_not_recommended,
        #         #                            weights=items_entropy[items_not_recommended]
        #         #                            ,k=1)[0]
        #         best_item = items_not_recommended[np.argmax(items_entropy[items_not_recommended])]
        #     top_iids.append(best_item)

        # top_popularity_iids = list(reversed(np.argsort(items_entropy)))[:self.get_iterations()]
        # top_entropy_iids = list(reversed(np.argsort(items_popularity)))[:self.get_iterations()]

        # correlation = scipy.stats.pearsonr(items_entropy,items_popularity)[0]

        # # top_iids = list(reversed(np.argsort(items_popplusent)))[:self.get_iterations()]

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
