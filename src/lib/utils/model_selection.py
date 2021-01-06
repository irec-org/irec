from copy import copy
def ModelSelection:
    pass

def TrainTestConsumption(ModelSelection):
    def __init__(self,train_size=0.8, test_consumes=1,crono=False):
        self.train_size=0.8
        self.test_consumes=0.8
        self.crono = crono
        
    def apply(self,dataset):
        data = dataset.data
        num_users = len(np.unique(data[0]))
        num_train_users = round(num_users*(self.train_size))
        num_test_users = int(num_users-num_train_users)
        data_df = pd.DataFrame(data)
        users_items_consumed=data_df.groupby(0).count().iloc[:,0]
        test_candidate_users=list(users_items_consumed[users_items_consumed>=self.test_consumes].to_dict().keys())
        if self.crono:
            users_start_time = data_df.groupby(0).min()[3].to_numpy()
            test_uids = np.array(list(test_candidate_users[list(reversed(np.argsort(users_start_time[test_candidate_users])))])[:num_test_users])
        else:
            test_uids = np.array(random.sample(test_candidate_users,k=num_test_users))
        train_uids = np.array(list(set(range(num_users))-set(test_uids)))

        data_isin_test_uids = np.isin(data[0],test_uids)

        train_dataset = copy(dataset)
        train_dataset.data = data[~data_isin_test_uids]
        test_dataset = copy(dataset)
        test_dataset.data = data[data_isin_test_uids]
        return train_dataset, test_dataset
        # return data[~data_isin_test_uids],data[data_isin_test_uids]