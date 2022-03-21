from typing import TypedDict

import numpy as np
import pandas as pd

from irec.environment.dataset import Dataset
from irec.environment.registry import FilterRegistry, SplitRegistry

DatasetType = TypedDict('DatasetType', {'path': str, 'random_seed': float, 'file_delimiter': str, 'skip_head': bool})
FilterUsersType = TypedDict('FilterUsersType', {'min_consumption': int, 'num_users': int})
FilterItemsType = TypedDict('FilterItemsType', {'min_ratings': int, 'num_items': int})
FilteringType = TypedDict('FilteringType', {'filter_users': FilterUsersType, 'filter_items': FilterItemsType})
SplittingType = TypedDict('SplittingType', {'strategy': str, 'train_size': float, 'test_consumes': int})


class Loader:

    def __init__(self,
                 dataset: DatasetType,
                 prefiltering: FilteringType,
                 splitting: SplittingType) -> None:
        """__init__.

        Args:
            dataset (DatasetType): info required by the dataset
            prefiltering (FilteringType): info required by the prefiltering
            splitting (SplittingType): info required by the Splitting
        """
        # dataset attributes
        self.dataset_path = dataset["path"]
        self.random_seed = dataset["random_seed"]
        self.delimiter = dataset["file_delimiter"]
        self.skip_rows = 1 if dataset["skip_head"] else 0
        # filtering attributes
        self.prefiltering = prefiltering
        # TODO: test_consumes must go to splitting group
        # splitting attributes
        self.test_consumes = splitting["test_consumes"]
        self.strategy = splitting["strategy"]
        self.train_size = splitting["train_size"]

    def _read(self):
        """
        Returns:
            The data read according to the parameters specified.
        """

        data = np.loadtxt(self.dataset_path,
                          delimiter=self.delimiter,
                          skiprows=self.skip_rows)
        return data

    def _filter(self,
                data: np.array) -> np.array:
        """
        Args:
            data: the array of data previously read
        Returns:
            The data filtered by the filters applied.
        """

        data_df = pd.DataFrame(data)
        for key, filters in self.prefiltering.items():
            for filter_method, value in filters.items():
                print(f"\t {filter_method}: {value}")
                data_df = getattr(FilterRegistry.get(key), filter_method)(data_df, value)

        return data_df.to_numpy()

    def _split(self,
               dataset: Dataset):
        """
        Args:
            dataset (Dataset): an object of the dataset class
        Returns:
            # TODO: define it better
        """

        num_train_users = round(dataset.num_users * self.train_size)
        num_test_users = int(dataset.num_users - num_train_users)
        data_df = pd.DataFrame(dataset.data)
        # Get the required strategy
        split_strategy = SplitRegistry.get(self.strategy)(
            test_consumes=self.test_consumes,
            train_size=self.train_size)
        # Apply it in the data
        test_uids = split_strategy.get_test_uids(data_df, num_test_users)
        train_test_processor = split_strategy.split_dataset(dataset, test_uids)
        return train_test_processor

    def process(self):

        # Read the data
        data = self._read()
        # Create dataset
        dataset = Dataset(data)
        dataset.reset_index()
        dataset.set_parameters()
        # Apply filters if they were defined
        if len(self.prefiltering) > 0:
            filtered_data = self._filter(dataset)
            # update dataset
            dataset = Dataset(filtered_data)
            dataset.reset_index()
            dataset.set_parameters()
        # Apply the split approach
        print(f"\nApplying splitting strategy: {self.strategy}\n")
        train_test_processor = self._split(dataset.data)
        return train_test_processor
