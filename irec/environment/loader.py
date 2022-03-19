from typing import TypedDict

import numpy as np
import pandas as pd

from irec.environment.dataset import Dataset
from irec.environment.registry import FilterRegistry, SplitRegistry

FilterUsers = TypedDict('FilterUsers', {'min_consumption': int, 'num_users': int})
FilterItems = TypedDict('FilterItems', {'min_ratings': int, 'num_items': int})
Filtering = TypedDict('Filtering', {'filter_users': FilterUsers, 'filter_items': FilterItems, 'test_consumes': int})
Splitting = TypedDict('Splitting', {'strategy': str, 'train_size': float})


class Loader:

    def __init__(self,
                 dataset_path: str,
                 prefiltering: Filtering,
                 splitting: Splitting,
                 random_seed: int) -> None:
        """__init__.

        Args:
            dataset_path (str): the path to your data
            prefiltering (Filtering): info required by the prefiltering
            splitting (Splitting): info required by the Splitting
            random_seed (int): random seed
        """
        self.dataset_path = dataset_path
        self.prefiltering = prefiltering
        self.test_consumes = prefiltering["test_consumes"]
        self.strategy = splitting["strategy"]
        self.train_size = splitting["train_size"]
        self.random_seed = random_seed

    def _read(self,
              delimiter: str = ",",
              skiprows: int = 1):

        data = np.loadtxt(self.dataset_path,
                          delimiter=delimiter,
                          skiprows=skiprows)
        dataset = Dataset(data)
        dataset.reset_index()
        dataset.set_parameters()
        return dataset

    def _filter(self, data):

        data_df = pd.DataFrame(data.data)
        for key, filters in self.prefiltering.items():
            for filter_method, value in filters.items():
                print(f"\t {filter_method}: {value}")
                data_df = getattr(FilterRegistry.get(key), filter_method)(data_df, value)
        return data_df.to_numpy()

    def _split(self, dataset):

        num_users = len(np.unique(dataset.data[:, 0]))
        num_train_users = round(num_users * self.train_size)
        num_test_users = int(num_users - num_train_users)
        data_df = pd.DataFrame(dataset.data)

        split_strategy = SplitRegistry.get(self.strategy)(
            test_consumes=self.test_consumes,
            train_size=self.train_size)

        test_uids = split_strategy.get_test_uids(data_df, num_test_users)
        train_test_processor = split_strategy.split_dataset(dataset, test_uids)
        return train_test_processor

    def process(self):

        # Read the data
        data = self._read()

        # Apply filters if they were defined
        # TODO: check the test_consumes used
        # filters = self.prefiltering
        # del filters["test_consumes"]
        # if len(filters) == 0: return data
        if len(self.prefiltering) > 0:
            filtered_data = self._filter()
        else:
            filtered_data = data

        dataset = Dataset(filtered_data)
        dataset.reset_index()
        dataset.set_parameters()

        # Apply the split approach
        print(f"\nApplying splitting strategy: {self.strategy}\n")
        self._split()
