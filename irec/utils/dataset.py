import pandas as pd
from irec import value_functions
import numpy as np
import irec.value_functions
import random
import scipy.sparse
import os
import numpy as np
import os
from copy import copy


def _si(x):
    du0 = np.sort(np.unique(x))
    ind0 = np.searchsorted(du0, x)
    return ind0


class DatasetPreprocessor:
    def __init__(self, name, dataset_descriptor, preprocessor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.dataset_descriptor = dataset_descriptor
        self.preprocessor = preprocessor


class Pipeline:
    def __init__(self, steps=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if steps is None:
            self.steps = []
        else:
            self.steps = steps

    def process(self, data):
        buf = data
        for element in self.steps:
            buf = element.process(buf)
        return buf


class DatasetDescriptor:
    def __init__(self, dataset_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_dir = dataset_dir


class Dataset:
    def __init__(
        self,
        data,
        num_total_users=None,
        num_total_items=None,
        num_users=None,
        num_items=None,
        rate_domain=None,
        uids=None,
    ):
        self.data = data
        self.num_users = num_users
        self.num_items = num_items
        self.rate_domain = rate_domain
        self.uids = uids
        self.num_total_users = num_total_users
        self.num_total_items = num_total_items

    def update_from_data(self):
        self.num_users = len(np.unique(self.data[:, 0]))
        self.num_items = len(np.unique(self.data[:, 1]))
        self.rate_domain = set(np.unique(self.data[:, 2]))
        self.uids = np.unique(self.data[:, 0]).astype(int)
        self.iids = np.unique(self.data[:, 1]).astype(int)
        self.max_uid = np.max(self.uids)
        self.max_iid = np.max(self.iids)
        self.mean_rating = np.mean(self.data[:, 2])
        self.min_rating = np.min(self.data[:, 2])
        self.max_rating = np.max(self.data[:, 2])

    def update_num_total_users_items(self):
        # self.num_total_users = self.num_users
        # self.num_total_items = self.num_items
        self.num_total_users = self.max_uid + 1
        self.num_total_items = self.max_iid + 1
        # print(self.num_total_users)
        # print(self.num_total_items)

        # self.consumption_matrix = scipy.sparse.csr_matrix((self.data[:,2],(self..data[:,0],self.train_dataset.data[:,1])),(self.train_dataset.users_num,self.train_dataset.items_num))


class TrainTestDataset:
    def __init__(self, train, test):
        self.train = train
        self.test = test


class DataProcessor:
    def __init__(self, *args, **kwargs):
        del args, kwargs


class TRTE(DataProcessor):
    def process(self, dataset_dir):
        # dataset_dir = dataset_descriptor.dataset_dir
        train_data = np.loadtxt(os.path.join(dataset_dir, "train.data"), delimiter="::")
        test_data = np.loadtxt(os.path.join(dataset_dir, "test.data"), delimiter="::")

        dataset = Dataset(np.vstack([train_data, test_data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        train_dataset = copy(dataset)
        train_dataset.data = train_data
        train_dataset.update_from_data()
        test_dataset = copy(dataset)
        test_dataset.data = test_data
        test_dataset.update_from_data()
        return TrainTestDataset(train=train_dataset, test=test_dataset)


class TRTEPopular(DataProcessor):
    def __init__(self, items_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_rate = items_rate

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        data = np.vstack((test_dataset.data, train_dataset.data))
        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        num_items_to_sample = int(self.items_rate * dataset.num_total_items)
        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items),
        )
        items_popularity = irec.value_functions.MostPopular.get_items_popularity(
            consumption_matrix
        )
        top_popular_items = np.argsort(items_popularity)[::-1][num_items_to_sample]
        test_dataset.data = test_dataset.data[
            test_dataset.data[:, 1].isin(top_popular_items)
        ]
        test_dataset.update_from_data()
        train_dataset.data = train_dataset.data[
            train_dataset.data[:, 1].isin(top_popular_items)
        ]
        train_dataset.update_from_data()

        # train_dataset.data[train_dataset.data[:,1].isin(top_popular_items)]

        # train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset


class TRTERandom(DataProcessor):
    def __init__(
        self, min_ratings, random_seed, probability_keep_item, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.min_ratings = min_ratings
        self.random_seed = random_seed
        self.probability_keep_item = probability_keep_item

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        # ttc = TrainTestConsumption(self.train_size, self.test_consumes,
        # self.crono, self.random_seed)
        # train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset


class MovieLens100k(DataProcessor):
    def process(self, dataset_dir):
        data = np.loadtxt(os.path.join(dataset_dir, "u.data"), delimiter="\t")
        data[:, 0] = data[:, 0] - 1
        data[:, 1] = data[:, 1] - 1
        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        return dataset


class DefaultDataset(DataProcessor):

    def filter_users(self, df_dataset, filters):
        if filters == None: return df_dataset
        def min_consumption(df_dataset, min_consumption):
            selected_users = dict(df_dataset.groupby(0)[1].agg("count")[lambda consumption: consumption >= min_consumption])
            return df_dataset[df_dataset[0].isin(selected_users)]

        def num_users(df_dataset, num_users):
            try: selected_users = random.sample(list(df_dataset[0].unique()), num_users)
            except: return df_dataset
            return df_dataset[df_dataset[0].isin(selected_users)]

        for filter_user in filters:
            df_dataset = eval(filter_user)(df_dataset, filters[filter_user])
 
        return df_dataset

    def filter_items(self, df_dataset, filters):

        def min_ratings(df_dataset, min_ratings):
            selected_items = dict(df_dataset.groupby(1)[0].agg("count")[lambda ratings: ratings >= min_ratings])
            return df_dataset[df_dataset[1].isin(selected_items)]

        def num_items(df_dataset, num_items):
            try: selected_items = random.sample(list(df_dataset[1].unique()), num_items)
            except: return df_dataset
            return df_dataset[df_dataset[1].isin(selected_items)]

        for filter_item in filters:
            df_dataset = eval(filter_item)(df_dataset, filters[filter_item])

        return df_dataset


    def prefiltering(self, ds, filters):

        del filters["test_consumes"]
        if len(filters) == 0: return ds
        data_df = pd.DataFrame(ds.data)

        print("Applying filters...")
        for key, filters in filters.items():
            print("\t", key, filters)
            data_df = eval(f"self.{key}")(data_df, filters)

        dataset = Dataset(data_df.to_numpy())
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        return dataset

    def process(self, dataset_dir):
        data = np.loadtxt(
            os.path.join(dataset_dir, "ratings.csv"), delimiter=",", skiprows=1
        )

        data[:, 0] = _si(data[:, 0])
        data[:, 1] = _si(data[:, 1])

        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        # print(max(dataset.uids), dataset.num_total_users)
        return dataset


class MovieLens1M(DataProcessor):
    def process(self, dataset_dir):
        data = np.loadtxt(os.path.join(dataset_dir, "ratings.dat"), delimiter="::")
        iids = dict()
        for i, iid in enumerate(np.unique(data[:, 1])):
            iids[iid] = i
        data[:, 1] = np.vectorize(lambda x: iids[x])(data[:, 1])
        data[:, 0] = data[:, 0] - 1
        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        return dataset


class MovieLens10M(DataProcessor):
    def process(self, dataset_dir):
        data = np.loadtxt(os.path.join(dataset_dir, "ratings.dat"), delimiter="::")
        iids = dict()
        for i, iid in enumerate(np.unique(data[:, 1])):
            iids[iid] = i
        data[:, 1] = np.vectorize(lambda x: iids[x])(data[:, 1])
        data[:, 0] = data[:, 0] - 1
        dataset = Dataset(data)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        return dataset


class Netflix:
    def _netflix_read_ratings(self, fileName):
        file = open(fileName, "r")
        file.readline()
        numratings = np.sum([1 for line in open(fileName)])
        usersId = np.zeros(numratings, dtype=np.int32)
        itemsId = np.zeros(numratings, dtype=np.int32)
        timestamp = np.zeros(numratings, dtype=np.uint64)
        ratings = np.zeros(numratings, dtype=np.uint8)

        file = open(fileName, "r")
        file.readline()
        cont = 0
        for row in file:
            values = row.split(",")
            uid, iid, rating, ts = (
                int(float(values[0])),
                int(float(values[1])),
                values[2],
                int(float(values[3].replace("\n", ""))),
            )
            usersId[cont] = uid
            itemsId[cont] = iid
            ratings[cont] = rating
            timestamp[cont] = ts
            cont += 1

        file.close()
        return usersId, itemsId, ratings, timestamp, numratings

    def process(self, dataset_dir):
        # base_dir = self.BASES_DIRS[self.base]
        # u_train, i_train, r_train, t_train, numr_train = _netflix_read_ratings(
        #     dataset_dir + "train.data"
        # )
        # u_test, i_test, r_test, t_test, numr_test = _netflix_read_ratings(
        #     dataset_dir + "test.data"
        # )
        # test_data = np.array((u_test, i_test, r_test, t_test))
        # train_data = np.array((u_train, i_train, r_train, t_train))

        usersId, itemsId, ratings, timestamp, numratings = self._netflix_read_ratings(
            dataset_dir + "ratings.csv"
        )
        dataset = np.array([usersId, itemsId, ratings, timestamp]).T
        # return train_data, test_data
        dataset = Dataset(dataset)
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        return dataset


class TrainTestConsumption(DataProcessor):
    def __init__(self, train_size, test_consumes, strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_size = train_size
        self.test_consumes = test_consumes
        self.strategy = strategy

    def splitting(self, strategy, data_df, num_test_users):

        users_items_consumed = data_df.groupby(0).count().iloc[:, 0]
        test_candidate_users = list(
            users_items_consumed[users_items_consumed >= self.test_consumes]
            .to_dict()
            .keys()
        )

        if strategy == "temporal":
            test_candidate_users = np.array(test_candidate_users, dtype=int)
            users_start_time = data_df.groupby(0).min()[3].to_numpy()
            test_uids = np.array(
                list(
                    test_candidate_users[
                        list(
                            reversed(np.argsort(users_start_time[test_candidate_users]))
                        )
                    ]
                )[:num_test_users]
            )
        elif strategy == "random":
            test_uids = np.array(random.sample(test_candidate_users, k=num_test_users))
        else:
            raise ValueError(f'The split strategy {strategy} does not exist', 'temporal', 'random') 

        train_uids = np.array(list(set(range(len(data_df[0].unique()))) - set(test_uids)))

        return train_uids, test_uids

    def process(self, ds):
        data = ds.data

        data[:, 0] = _si(data[:, 0])
        data[:, 1] = _si(data[:, 1])

        ds = Dataset(data)
        ds.update_from_data()
        ds.update_num_total_users_items()

        num_users = len(np.unique(data[:, 0]))
        num_train_users = round(num_users * (self.train_size))
        num_test_users = int(num_users - num_train_users)
        data_df = pd.DataFrame(data)

        train_uids, test_uids = self.splitting(self.strategy, data_df, num_test_users)
  
        data_isin_test_uids = np.isin(data[:, 0], test_uids)

        train_dataset = copy(ds)
        train_dataset.data = data[~data_isin_test_uids, :]
        ds.update_from_data()
        test_dataset = copy(ds)
        test_dataset.data = data[data_isin_test_uids, :]
        ds.update_from_data()
        print("Test shape:", test_dataset.data.shape)
        print("Train shape:", train_dataset.data.shape)
        return TrainTestDataset(train=train_dataset, test=test_dataset)


class TRTETrainValidation(DataProcessor):
    def __init__(self, train_size, test_consumes, crono, random_seed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_size = train_size
        self.test_consumes = test_consumes
        self.crono = crono
        self.random_seed = random_seed

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        ttc = TrainTestConsumption(
            self.train_size, self.test_consumes, self.crono, self.random_seed
        )
        train_dataset, test_dataset = ttc.process(train_dataset)
        return train_dataset, test_dataset


class TRTESample(DataProcessor):
    def __init__(self, items_rate, sample_method, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items_rate = items_rate
        self.sample_method = sample_method

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        dataset = Dataset(np.vstack([train_dataset.data, test_dataset.data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items),
        )
        num_items_to_sample = int(self.items_rate * dataset.num_total_items)
        if self.sample_method == "entropy":
            items_values = irec.value_functions.Entropy.Entropy.get_items_entropy(
                consumption_matrix
            )
        elif self.sample_method == "popularity":
            items_values = (
                irec.value_functions.MostPopular.MostPopular.get_items_popularity(
                    consumption_matrix
                )
            )

        best_items = np.argpartition(items_values, -num_items_to_sample)[
            -num_items_to_sample:
        ]
        dataset.data = dataset.data[np.isin(dataset.data[:, 1], best_items), :]

        new_iids = dict()
        for i, iid in enumerate(np.unique(dataset.data[:, 1])):
            new_iids[iid] = i
        for i in range(len(dataset.data)):
            dataset.data[i, 1] = new_iids[dataset.data[i, 1]]

        dataset.update_from_data()
        dataset.update_num_total_users_items()

        train_uids = train_dataset.uids
        test_uids = test_dataset.uids

        train_dataset = copy(dataset)
        train_dataset.data = dataset.data[np.isin(dataset.data[:, 0], train_uids)]
        train_dataset.update_from_data()

        test_dataset = copy(dataset)
        test_dataset.data = dataset.data[np.isin(dataset.data[:, 0], test_uids)]
        test_dataset.update_from_data()
        return train_dataset, test_dataset


class PopularityFilter(DataProcessor):
    def __init__(self, keep_popular, num_items_threshold, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_popular = keep_popular
        self.num_items_threshold = num_items_threshold

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        dataset = Dataset(np.vstack([train_dataset.data, test_dataset.data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items),
        )
        # num_items_to_sample = int(self.items_rate * dataset.num_total_items)
        items_values = irec.value_functions.MostPopular.get_items_popularity(
            consumption_matrix
        )
        items_sorted = np.argsort(items_values)[::-1]
        if self.keep_popular:
            items_to_keep = items_sorted[: self.num_items_threshold]
        else:
            items_to_keep = items_sorted[self.num_items_threshold :]

        dataset.data = dataset.data[np.isin(dataset.data[:, 1], items_to_keep), :]

        new_iids = dict()
        for i, iid in enumerate(np.unique(dataset.data[:, 1])):
            new_iids[iid] = i
        for i in range(len(dataset.data)):
            dataset.data[i, 1] = new_iids[dataset.data[i, 1]]
        new_uids = dict()
        for i, uid in enumerate(np.unique(dataset.data[:, 0])):
            new_uids[uid] = i
        for i in range(len(dataset.data)):
            dataset.data[i, 0] = new_uids[dataset.data[i, 0]]

        dataset.update_from_data()
        dataset.update_num_total_users_items()

        # train_uids = train_dataset.uids
        # test_uids = test_dataset.uids

        # train_dataset = copy(dataset)
        # train_dataset.data = dataset.data[np.isin(dataset.data[:, 0],
        # train_uids)]
        # train_dataset.update_from_data()

        # test_dataset = copy(dataset)
        # test_dataset.data = dataset.data[np.isin(dataset.data[:, 0], test_uids)]
        # test_dataset.update_from_data()
        return dataset


class CombineTrainTest(DataProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        dataset = Dataset(np.vstack([train_dataset.data, test_dataset.data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        return dataset


class PopRemoveEnt(DataProcessor):
    def __init__(self, num_items_threshold, new_rating, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.keep_popular = keep_popular
        self.num_items_threshold = num_items_threshold
        self.new_rating = new_rating

    def process(self, train_dataset_and_test_dataset):
        train_dataset = train_dataset_and_test_dataset[0]
        test_dataset = train_dataset_and_test_dataset[1]
        dataset = Dataset(np.vstack([train_dataset.data, test_dataset.data]))
        dataset.update_from_data()
        dataset.update_num_total_users_items()
        consumption_matrix = scipy.sparse.csr_matrix(
            (dataset.data[:, 2], (dataset.data[:, 0], dataset.data[:, 1])),
            (dataset.num_total_users, dataset.num_total_items),
        )
        # num_items_to_sample = int(self.items_rate * dataset.num_total_items)
        items_values = irec.value_functions.MostPopular.get_items_popularity(
            consumption_matrix
        )
        items_sorted = np.argsort(items_values)[::-1]
        # if self.keep_popular:
        items_to_keep = items_sorted[: self.num_items_threshold]
        # else:
        # items_to_keep = items_sorted[self.num_items_threshold:]
        train_dataset.data[
            np.isin(train_dataset.data[:, 1], items_to_keep), 2
        ] = self.new_rating
        test_dataset.data[
            np.isin(test_dataset.data[:, 1], items_to_keep), 2
        ] = self.new_rating
        # dataset.update_from_data()
        # dataset.update_num_total_users_items()
        return train_dataset, test_dataset
