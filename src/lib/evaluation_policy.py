from collections import defaultdict
import scipy.sparse
def EvaluationPolicy:
    pass

class Interaction(EvaluationPolicy):
    def __init__(self,parameters,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters

    def evaluate(self,model,train_data,test_data):
        test_users = np.unique(test_data[0])
        # total_num_users = len(np.unique(np.concatenate((train_data[0],
        #                                             test_data[0]),axis=None)))

        # total_num_items = len(np.unique(np.concatenate((train_data[1],
        #                                             test_data[1]),axis=None)))
        
        test_consumption_matrix = scipy.sparse.csr_matrix((test_data[2],(test_data[0],test_data[1])))
        users_items_recommended = defaultdict(list)
        num_test_users = len(test_users)
        model.train(train_data)
        users_num_interactions = defaultdict(int)
        available_users = set(test_users)
        for i in range(num_test_users*self.parameters.interactions):
            uid = random.sample(available_users,k=1)[0]
            # for i in range(self.parameters.interaction_size):
            not_recommended = np.ones(num_items,dtype=bool)
            not_recommended[users_items_recommended[uid]] = 0
            items_not_recommended = np.nonzero(not_recommended)[0]
            items_score = model.predict(uid,items_not_recommended)
            best_items = list(reversed(np.argsort(items_score)))[:self.parameters.interaction_size]
            # best_item = items_not_recommended[np.argmax(items_score)]
            users_items_recommended[uid].extend(best_items)

            # for i in range(self.parameters.interaction_size):
            for item in users_items_recommended[users_num_interactions[uid]*self.parameters.interaction_size:(users_num_interactions[uid]+1)*self.parameters.interaction_size]:
                model.update(uid,item,test_consumption_matrix[uid,item])

            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.parameters.interactions:
                available_users = available_users - {uid}
