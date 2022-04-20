from typing import Tuple, TypedDict
import pandas as pd
import numpy as np
import random

from irec.environment.dataset import Dataset
from irec.environment.registry import FilterRegistry, SplitRegistry

DatasetType = TypedDict('DatasetType', {'path': str, 'random_seed': float, 'file_delimiter': str, 'skip_head': bool})
FilterUsersType = TypedDict('FilterUsersType', {'min_consumption': int, 'num_users': int})
FilterItemsType = TypedDict('FilterItemsType', {'min_ratings': int, 'num_items': int})
FilteringType = TypedDict('FilteringType', {'filter_users': FilterUsersType, 'filter_items': FilterItemsType})
SplittingType = TypedDict('SplittingType', {'strategy': str, 'train_size': float, 'test_consumes': int})


class DefaultLoader:

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
        if "path" in dataset.keys():
            self.dataset_path = dataset["path"]
        else:
            raise IndexError("Dataset 'path' not exists. You must define your dataset path to be reader by the system.")

        self.random_seed = dataset["random_seed"] if "random_seed" in dataset.keys() else 0
        self.delimiter = dataset["file_delimiter"] if "file_delimiter" in dataset.keys() else ","
        self.skip_rows = int(dataset["skip_head"]) if "skip_head" in dataset.keys() else 1

        # filtering attributes
        self.prefiltering = prefiltering

        # splitting attributes
        self.test_consumes = splitting["test_consumes"] if "test_consumes" in splitting.keys() else 0
        self.strategy = splitting["strategy"] if "strategy" in splitting.keys() else "random"
        self.train_size = splitting["train_size"] if "train_size" in splitting.keys() else 0.8

    def _read(self) -> np.ndarray:
        """_read
            The data read according to the parameters specified.
            The expected columns are userId, itemId, rating, timestamp
        Returns:
            data (np.ndarray): The data loaded
        """
        data = np.loadtxt(self.dataset_path,
                          delimiter=self.delimiter,
                          skiprows=self.skip_rows)

        # TODO: check if timestamp exists. If not, create a sequential value for it

        return data

    def _filter(self,
                data: np.array) -> np.ndarray:
        """_filter

            Applies all filters specified in dataset_loaders.yaml

        Args:
            data: the array of data previously read

        Returns:
            data_df (np.array): The data filtered by the filters applied.
        """
        data_df = pd.DataFrame(data, columns=["userId", "itemId", "rating", "timestamp"])
        print(f"\nApplying filters...")
        for key, filters in self.prefiltering.items():
            print(f"{key}:")
            for filter_method, value in filters.items():
                print(f"\t {filter_method}: {value}")
                data_df = getattr(FilterRegistry.get(key), filter_method)(data_df, value)
    
        return data_df.to_numpy()

    def _split(self,
               dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """split

            Splits the data set into training and testing

        Args:
            dataset (Dataset): an object of the dataset class

        Returns:
            train_dataset (Dataset): the train
            test_dataset (Dataset): the test
        """
        num_train_users = round(dataset.num_users * self.train_size)
        num_test_users = int(dataset.num_users - num_train_users)
        # Get the required strategy
        split_strategy = SplitRegistry.get(self.strategy)(
            test_consumes=self.test_consumes,
            train_size=self.train_size)
        # Apply it in the data
        test_uids = split_strategy.get_test_uids(dataset.data, num_test_users)
        train_dataset, test_dataset = split_strategy.split_dataset(dataset.data, test_uids)
        train_dataset.update_num_total_users_items(
            num_total_users=dataset.num_total_users, 
            num_total_items=dataset.num_total_items
        )
        test_dataset.update_num_total_users_items(
            num_total_users=dataset.num_total_users, 
            num_total_items=dataset.num_total_items
        )
        return train_dataset, test_dataset

    def process(self) -> Tuple[Dataset, Dataset]:

        """process

        Perform complete processing of the dataset: read -> filter (optional) -> split

        Return: 
            train_dataset (Dataset): the train
            test_dataset (Dataset): the test
        """

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        # Read the data
        data = self._read()
        # Create dataset
        dataset = Dataset(data)
        dataset.reset_index()
        dataset.set_parameters()
        dataset.update_num_total_users_items()

        # Apply filters if they were defined
        if self.prefiltering != "None":
            filtered_data = self._filter(dataset.data)
            # update dataset
            dataset = Dataset(filtered_data)
            dataset.reset_index()
            dataset.set_parameters()
            dataset.update_num_total_users_items()

        # Create train and test set
        print(f"\nApplying splitting strategy: {self.strategy}\n")
        train_dataset, test_dataset = self._split(dataset)

        return train_dataset, test_dataset
