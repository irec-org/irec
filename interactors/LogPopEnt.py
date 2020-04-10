import numpy as np
from tqdm import tqdm
from . import Interactor, Entropy, MostPopular
import matplotlib.pyplot as plt
import scipy.stats
import os

class LogPopEnt(Interactor):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def interact(self, uids):
        super().interact()

        items_entropy = Entropy.get_items_entropy(self.consumption_matrix,uids)
        items_popularity = MostPopular.get_items_popularity(self.consumption_matrix,uids)
        
        items_logpopent = items_entropy * np.ma.log(items_popularity).filled(0)

        correlation = scipy.stats.pearsonr(items_entropy,items_popularity)[0]
        fig, ax = plt.subplots()
        ax.scatter(items_entropy,items_popularity,marker="D",color='darkblue')
        ax.set_ylabel("Popularity")
        ax.set_xlabel("Entropy")
        ax.text(0.3, 0.9 , f'Correlation coefficient: {correlation:.2f}', color='k',
                ha='center', va='center',
                bbox=dict(facecolor='none', edgecolor='k', pad=10.0),
                transform = ax.transAxes)
        fig.savefig(os.path.join(self.DIRS['img'],"corr_popent_"+self.get_name()+".png"))

        top_iids = list(reversed(np.argsort(items_logpopent)))[:self.get_iterations()]

        num_users = len(uids)
        for idx_uid in tqdm(range(num_users)):
            uid = uids[idx_uid]
            self.result[uid].extend(top_iids)
        self.save_result()
