import numpy as np
from tqdm import tqdm
from . import ValueFunction, Entropy, MostPopular, LogPopEnt, PopPlusEnt, ExperimentalValueFunction
import matplotlib.pyplot as plt
import scipy.stats
import os


class PPELPE(ExperimentalValueFunction):
    @staticmethod
    def get_items_ppelpe(items_popularity, items_entropy, do_sum=True):
        items_popplusent = PopPlusEnt.get_items_popplusent(
            items_popularity, items_entropy)
        items_logpopent = LogPopEnt.get_items_logpopent(
            items_popularity, items_entropy)
        if do_sum:
            res = items_popplusent + items_logpopent
        else:
            res = items_popplusent * items_logpopent
        return res / np.max(res)

    def reset(self, observation):
        train_dataset = observation
        super().reset(train_dataset)

        items_entropy = Entropy.get_items_entropy(
            self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(
            self.train_consumption_matrix, normalize=False)
        self.items_ppelpe = self.get_items_ppelpe(items_popularity,
                                                  items_entropy)

    def action_estimates(self, candidate_actions):
        uid = candidate_actions[0]
        candidate_items = candidate_actions[1]
        items_score = self.items_ppelpe[candidate_items]
        return items_score, None

        # correlation = scipy.stats.pearsonr(items_entropy,items_popularity)[0]

        # top_iids = list(reversed(np.argsort(items_ppelpe)))[:self.get_iterations()]

        # fig, ax = plt.subplots()
        # ax.scatter(items_entropy,items_popularity,marker="D",color='darkblue')
        # ax.set_ylabel("Popularity")
        # ax.set_xlabel("Entropy")
        # ax.text(0.3, 0.9 , f'Correlation coefficient: {correlation:.2f}', color='k',
        #         ha='center', va='center',
        #         bbox=dict(facecolor='none', edgecolor='k', pad=10.0),
        #         transform = ax.transAxes)
        # for start, end, color in [(0,10,'green'),(10,20,'red'),(20,30,'darkred'),(30,40,'yellow'),(40,50,'orange')]:
        #     print(top_iids[start:end])
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
