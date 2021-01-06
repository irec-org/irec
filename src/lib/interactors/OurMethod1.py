import numpy as np
from tqdm import tqdm
import util
import interactors
from threadpoolctl import threadpool_limits
import ctypes
import scipy.spatial
import matplotlib.pyplot as plt
import os
import pickle
class OurMethod1(interactors.Interactor):
    def __init__(self, alpha=1.0, stop=None, weight_method='change',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.weight_method = weight_method
        self.stop = stop

    def interact(self, items_latent_factors):
        super().interact()
        uids = self.test_users

        self.items_latent_factors = items_latent_factors
        num_users = len(uids)

        items_entropy = interactors.Entropy.get_items_entropy(self.train_consumption_matrix)
        items_popularity = interactors.MostPopular.get_items_popularity(self.train_consumption_matrix,normalize=False)
        # self.items_bias = interactors.PPELPE.get_items_ppelpe(items_popularity,items_entropy)
        self.items_bias = interactors.LogPopEnt.get_items_logpopent(items_popularity,items_entropy)

        assert(self.items_bias.min() >= 0 and self.items_bias.max() == 1)

        self_id = id(self)
        np.seterr('warn')
        with threadpool_limits(limits=1, user_api='blas'):
            args = [(self_id,int(uid),) for uid in uids]
            results = util.run_parallel(self.interact_user,args)

        if self.exit_when_consumed_all:
            for i, user_result in enumerate(results):
                if not self.results_save_relevants:
                    self.results[uids[i]] = user_result
                else:
                    self.results[uids[i]] = user_result[np.isin(user_result,np.nonzero(self.test_consumption_matrix[uids[i]].A.flatten())[0])]
        else:
            users_global_model_weights = dict()

            for i, (user_result, global_model_weights) in enumerate(results):
                if not self.results_save_relevants:
                    self.results[uids[i]] = user_result
                else:
                    self.results[uids[i]] = user_result[np.isin(user_result,np.nonzero(self.test_consumption_matrix[uids[i]].A.flatten())[0])]
                users_global_model_weights[uids[i]] = global_model_weights

            users_global_model_weights=np.array(list(users_global_model_weights.values()))


            fig, ax = plt.subplots()
            ax.errorbar(np.arange(users_global_model_weights.shape[1])+1,
                        y=np.mean(users_global_model_weights,axis=0),
                        yerr=np.std(users_global_model_weights,axis=0), label= '$\overline{x}$ and $s$',marker='.',color='k',
                        capsize=3)
            ax.plot(np.arange(users_global_model_weights.shape[1])+1,
                    np.min(users_global_model_weights,axis=0), label='Min weight',
                    color='green'
            )
            ax.plot(np.arange(users_global_model_weights.shape[1])+1,
                    np.max(users_global_model_weights,axis=0), label='Max weight',
                    color='red',
            )
            ax.set_xlabel("Interaction $t$")
            ax.set_ylabel("Weight $\Phi_{t}$")
            ax.set_xlim(1,users_global_model_weights.shape[1])
            ax.legend()
            fig.savefig(os.path.join(self.DIRS['img'],"weights_"+self.get_name()+".png"))
            with open(os.path.join(self.DIRS['result'],"weights_"+self.get_name()+".pickle"),'wb') as f:
                pickle.dump(users_global_model_weights,f)

        self.save_results()

    def get_global_model_weight(self,user_latent_factors_history,num_correct_items_history,distance_history):
        if self.weight_method == 'stop':
            b = self.stop
            a = min(np.sum(num_correct_items_history),b)
            return (1-np.round(pow(2,a)/pow(2,b),3))
        elif self.weight_method == 'change':
            if len(user_latent_factors_history) == 0:
                return 1
            times_with_reward = np.nonzero(num_correct_items_history)[0]
            if len(times_with_reward) < 2:
                return 1
            # print(user_latent_factors_history[times_with_reward][-1])
            # print(user_latent_factors_history[times_with_reward][-2])
            res = scipy.spatial.distance.cosine(user_latent_factors_history[times_with_reward][-1],user_latent_factors_history[times_with_reward][-2])
            distance_history.append(res)
            res = res/np.max(distance_history)
            # res = (res+1)/2
            return res
        else:
            raise RuntimeError

    @staticmethod
    def interact_user(obj_id,uid):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        num_lat = len(self.items_latent_factors[0])
        I = np.eye(num_lat)

        user_candidate_items = np.array(list(range(len(self.items_latent_factors))))
        num_user_candidate_items = len(user_candidate_items)
        b = np.zeros(num_lat)
        A = I.copy()
        result = []
        # old_mean = np.ones(b.shape)*1
        user_latent_factors_history = np.empty(shape=(0, num_lat))
        num_correct_items_history = [0]
        similarity_score = [0]
        distance_history = []
        global_model_weights = []
        num_correct_items = 0
        for i in range(self.interactions):
            
            user_latent_factors = np.dot(np.linalg.inv(A),b)
            user_latent_factors_history = np.vstack([user_latent_factors_history,user_latent_factors])
            global_model_weight = self.get_global_model_weight(user_latent_factors_history,
                                                               num_correct_items_history,
                                                               distance_history)
            global_model_weights.append(global_model_weight)
            # if uid == 2:
            #     print(np.sum(num_correct_items_history))
            #     print(global_model_weight)
            #     times_with_reward = np.nonzero(num_correct_items_history)[0]
            #     print(times_with_reward)
                
            #     if len(times_with_reward) >= 2:
            #         print(user_latent_factors_history[times_with_reward][-1])
            #         print(user_latent_factors_history[times_with_reward][-2])
                # print(num_correct_items_history)
                # print(mean)
                # print(np.linalg.norm(mean-old_mean))
                # print(np.corrcoef(mean,old_mean))
                # print(mean,old_mean)
                # print("%d %.5f %d"%(i,scipy.spatial.distance.cosine(mean,old_mean),np.count_nonzero([self.get_reward(uid,item) for item in result])))
            items_uncertainty = np.sqrt(np.sum(self.items_latent_factors[user_candidate_items].dot(np.linalg.inv(A)) * self.items_latent_factors[user_candidate_items],axis=1))
            items_user_similarity = user_latent_factors @ self.items_latent_factors[user_candidate_items].T
            user_model_items_score = items_user_similarity + self.alpha*items_uncertainty
            global_model_items_score = self.items_bias[user_candidate_items]
            user_model_items_score_min = np.min(user_model_items_score)
            user_model_items_score_max = np.max(user_model_items_score)
            if user_model_items_score_max-user_model_items_score_min != 0:
                global_model_items_score = global_model_items_score*(user_model_items_score_max-user_model_items_score_min) + user_model_items_score_min

            items_score =  (1-global_model_weight)*user_model_items_score + global_model_weight*global_model_items_score

            best_items = user_candidate_items[np.argsort(items_score)[::-1]][:self.interaction_size]
            user_candidate_items = user_candidate_items[~np.isin(user_candidate_items,best_items)]
            result.extend(best_items)

            num_correct_items_history.append(0)
            for max_i in result[i*self.interaction_size:(i+1)*self.interaction_size]:
                max_item_latent_factors = self.items_latent_factors[max_i]
                A += max_item_latent_factors[:,None].dot(max_item_latent_factors[None,:])
                num_user_candidate_items -= 1
                if self.get_reward(uid,max_i) >= self.threshold:
                    b += self.get_reward(uid,max_i)*max_item_latent_factors
                    num_correct_items_history[-1] += 1
                    num_correct_items += 1
                    if self.exit_when_consumed_all and num_correct_items == self.users_num_correct_items[uid]:
                        print(f"Exiting user {uid} with {len(result)} items in total and {num_correct_items} correct ones")
                        return np.array(result)

                    
            # old_mean = mean.copy()

        if self.exit_when_consumed_all:
            return np.array(result)
        else:
            return np.array(result), global_model_weights