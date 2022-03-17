from posixpath import split
import random
from typing import Any, Dict, List, TypedDict
from mergedeep import Strategy
import numpy as np
import pandas as pd

from utils import dataset as dataset_module

Splitting = TypedDict('splitting', {'strategy': str, 'train_size': float})
FilterUsers = TypedDict('filter_users', {'min_consumption': int, 'num_users': int})
FilterItems = TypedDict('filter_items', {'min_ratings': int, 'num_items': int})
Prefiltering = TypedDict('prefiltering', {'filter_users': FilterUsers, 'filter_items': FilterItems})

class DatasetLoader:

    def load(self):
        pass

class DefaultDatasetLoader:
    """DefaultDatasetLoader.
    """
    def __init__(self, 
                 dataset_path: str, 
                 prefiltering: Prefiltering, 
                 splitting: Splitting, 
                 random_seed: int) -> None:
        """__init__.

        Args:
            dataset_path: (str): 
            prefiltering (Prefiltering): info required by the Prefiltering
            splitting (Splitting): info required by the Splitting
            random_seed (int): randoom seed
        """
        self.dataset_path = dataset_path
        self.prefiltering = prefiltering
        self.test_consumes = prefiltering["test_consumes"]
        self.strategy = splitting["strategy"]
        self.train_size = splitting["train_size"]
        self.random_seed = random_seed

    def apply_filters(self, data):
        filters = self.prefiltering
        del filters["test_consumes"]
        if len(filters) == 0: return data

        dict_filters = {"filter_users": "FilteringByUsers",
            "filter_items": "FilteringByItems"}
        data_df = pd.DataFrame(data.data)
        
        print("Applying filters...")
        for key, filters in filters.items():
            filter_name = dict_filters[key]
            exec(f"from environment.filtering import {filter_name}")
            for filter_method, value in filters.items():
                print(f"\t {filter_method}: {value}")
                data_df = eval(f"{filter_name}.{filter_method}")(data_df, value)

        filtered_dataset = dataset_module.Dataset(data_df.to_numpy())
        filtered_dataset.set_parameters()
        return filtered_dataset

    def apply_splitting(self, dataset):
        dict_split = {"temporal": "Temporal",
            "random": "Random"}
        
        split_name = dict_split[self.strategy]
        data = dataset.data
        data[:, 0] = dataset_module._si(data[:, 0])
        data[:, 1] = dataset_module._si(data[:, 1])
        dataset = dataset_module.Dataset(data)
        dataset.set_parameters()

        num_users = len(np.unique(data[:, 0]))
        num_train_users = round(num_users * (self.train_size))
        num_test_users = int(num_users - num_train_users)
        data_df = pd.DataFrame(data)
        
        exec(f"from environment.splitting import {split_name}")
        split_strategy = eval(split_name)(
            strategy=self.strategy, 
            test_consumes=self.test_consumes,
            train_size=self.train_size)

        test_uids = split_strategy.get_test_uids(data_df, num_test_users)
        traintest_processor = split_strategy.split_dataset(dataset, test_uids)
        return traintest_processor

    def load(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        default_processor = dataset_module.DefaultDataset()
        data = default_processor.process(self.dataset_path)
        data = self.apply_filters(data)
        traintest_processor = self.apply_splitting(data)
        return traintest_processor


class DefaultValidationDatasetLoader:
    def __init__(self, dataset_path, prefiltering, splitting , random_seed) -> None:
        self.dataset_path = dataset_path
        self.prefiltering = prefiltering
        self.test_consumes = prefiltering["test_consumes"]
        self.strategy = splitting["strategy"]
        self.train_size = splitting["train_size"]
        self.random_seed = random_seed

    def load(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        default_processor = dataset_module.DefaultDataset()
        data = default_processor.process(self.dataset_path)
        data = default_processor.prefiltering(data, self.prefiltering)

        traintest_processor = dataset_module.TrainTestConsumption(
            strategy=self.strategy,
            test_consumes=self.test_consumes,
            train_size=self.train_size,
        )
        res = traintest_processor.process(traintest_processor.process(data).train)

        return res


class TRTEDatasetLoader:
    def __init__(
        self,
        dataset_path
        # self, dataset_path, crono, random_seed, test_consumes, train_size
    ) -> None:
        self.dataset_path = dataset_path
        # self.crono = crono
        # self.random_seed = random_seed
        # self.test_consumes = test_consumes
        # self.train_size = train_size

    def load(self):
        # np.random.seed(self.random_seed)
        trte_processor = dataset_module.TRTE()
        data = trte_processor.process(self.dataset_path)
        res = data
        # traintest_processor = dataset_module.TrainTestConsumption(
        # crono=self.crono,
        # test_consumes=self.test_consumes,
        # train_size=self.train_size,
        # )
        # res = traintest_processor.process(data)
        return res


class TRTEValidationDatasetLoader:
    def __init__(
        self, dataset_path, crono, random_seed, test_consumes, train_size
    ) -> None:
        self.dataset_path = dataset_path
        self.crono = crono
        self.random_seed = random_seed
        self.test_consumes = test_consumes
        self.train_size = train_size

    def load(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        trte_processor = dataset_module.TRTE()
        data = trte_processor.process(self.dataset_path)

        traintest_processor = dataset_module.TrainTestConsumption(
            crono=self.crono,
            test_consumes=self.test_consumes,
            train_size=self.train_size,
        )
        res = traintest_processor.process(data.train)

        return res
