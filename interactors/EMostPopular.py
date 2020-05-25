import numpy as np
from tqdm import tqdm
from . import Interactor, Entropy,Entropy0, MostPopular,LogPopEnt
import matplotlib.pyplot as plt
import scipy.stats
import os
import random

class EMostPopular(Interactor):
    def __init__(self,epsilon=0.2,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def top_emostpopular(self,items_exploitation,items_exploration):
        top_iids = []
        num_items = self.train_consumption_matrix.shape[1]
        for i in range(num_items):
            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[top_iids] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            if self.epsilon < np.random.rand():
                best_item = items_not_recommended[np.argmax(items_exploitation[items_not_recommended])]
            else:
                # best_item = random.choices(items_not_recommended,
                #                            weights=items_entropy[items_not_recommended]
                #                            ,k=1)[0]
                best_item = items_not_recommended[np.argmax(items_exploration[items_not_recommended])]
            top_iids.append(best_item)
        return top_iids

    def interact(self):
        super().interact()
        uids = self.test_users
        items_entropy = Entropy.get_items_entropy(self.train_consumption_matrix)
        # items_entropy0 = Entropy0.get_items_entropy(self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)

        # items_probability = items_entropy/items_entropy.sum()
        top_iids = self.top_emostpopular(items_popularity,items_entropy)
        # top_iids = []
        # num_items = self.train_consumption_matrix.shape[1]
        # for i in range(num_items):
        #     not_recommended = np.ones(num_items,dtype=bool)
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

        correlation = scipy.stats.pearsonr(items_entropy,items_popularity)[0]

        # top_iids = list(reversed(np.argsort(items_popplusent)))[:self.get_iterations()]

        fig, ax = plt.subplots()
        ax.scatter(items_entropy,items_popularity,marker="D",color='darkblue')
        ax.set_ylabel("Popularity")
        ax.set_xlabel("Entropy")
        ax.text(0.3, 0.9 , f'Correlation coefficient: {correlation:.2f}', color='k',
                ha='center', va='center',
                bbox=dict(facecolor='none', edgecolor='k', pad=10.0),
                transform = ax.transAxes)
        for start, end, color in [(0,10,'green'),(10,20,'red'),(20,30,'darkred'),(30,40,'yellow'),(40,50,'orange')]:
            ax.scatter(items_entropy[top_iids[start:end]],items_popularity[top_iids[start:end]],marker='D',color=color)
            print("[%d,%d] sum(popularity)=%.2f sum(entropy)=%.2f"%(start,end,
                                                                 np.sum(items_popularity[top_iids[start:end]]/np.max(items_popularity)),
                                                                 np.sum(items_entropy[top_iids[start:end]]/np.max(items_entropy))))
        fig.savefig(os.path.join(self.DIRS['img'],"corr_popent_"+self.get_name()+".png"))


        num_users = len(uids)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.results[uid].extend(top_iids)
        self.save_results()