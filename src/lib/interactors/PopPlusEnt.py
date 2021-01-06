import numpy as np
from tqdm import tqdm
from . import Interactor, Entropy, MostPopular,LogPopEnt
import matplotlib.pyplot as plt
import scipy.stats
import os

class PopPlusEnt(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_items_popplusent(items_popularity,items_entropy,log=False):
        if not log:
            items_popplusent = items_entropy/np.max(items_entropy) + items_popularity/np.max(items_popularity)
        else:
            items_popplusent = items_entropy/np.max(items_entropy) + np.ma.log(items_popularity).filled(0)/np.max(np.ma.log(items_popularity).filled(0))
        return items_popplusent/np.max(items_popplusent)

    def interact(self):
        super().interact()
        uids = self.test_users
        items_entropy = Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
        items_popplusent = PopPlusEnt.get_items_popplusent(items_popularity,items_entropy)

        correlation = scipy.stats.pearsonr(items_entropy,items_popularity)[0]

        top_iids = list(reversed(np.argsort(items_popplusent)))[:self.get_iterations()]
        fig, ax = plt.subplots()
        ax.hist(items_entropy,color='k')
        ax.set_xlabel("$Pop + Ent$")
        ax.set_ylabel("#Items")
        fig.savefig(os.path.join(self.DIRS['img'],"popplusent_"+self.get_name()+".png"))

        fig, ax = plt.subplots()
        ax.scatter(items_entropy,items_popularity,marker="D",color='darkblue')
        ax.set_ylabel("Popularity")
        ax.set_xlabel("Entropy")
        ax.text(0.3, 0.9 , f'Correlation coefficient: {correlation:.2f}', color='k',
                ha='center', va='center',
                bbox=dict(facecolor='none', edgecolor='k', pad=10.0),
                transform = ax.transAxes)
        for start, end, color in [(0,10,'green'),(10,20,'red'),(20,30,'darkred'),(30,40,'yellow'),(40,50,'orange')]:
            print(top_iids[start:end])
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