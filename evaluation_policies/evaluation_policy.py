def EvaluationPolicy:
    def __init__(self,interactions=100,interaction_size=1):
        pass

class Interaction(EvaluationPolicy):
    def __init__(self,interactions,interaction_size,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.interactions = interactions
        self.interaction_size = interaction_size

    def evaluate(self,model,train_data,test_data):
        users_items_recommended = defaultdict(list)
        num_test_users = len(uids)
        model.train(train_data)
        users_num_interactions = defaultdict(int)
        available_users = set(uids)
        for i in range(num_test_users*self.interactions):
            uid = random.sample(available_users,k=1)[0]
            for i in range(self.interaction_size):
                not_recommended = np.ones(num_items,dtype=bool)
                not_recommended[users_items_recommended[uid]] = 0
                items_not_recommended = np.nonzero(not_recommended)[0]
                best_item = model.predict(uid,items_not_recommended)
                users_items_recommended[uid].append(best_item)

            users_num_interactions[uid] += 1
            if users_num_interactions[uid] == self.interactions:
                available_users = available_users - {uid}

            model.update_parameters(uid)
