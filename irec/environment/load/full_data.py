from typing import TypedDict
import pandas as pd
import numpy as np
import random

from irec.environment.dataset import Dataset
from irec.environment.dataset import TrainTestDataset
from irec.environment.registry import FilterRegistry, SplitRegistry

# TODO: change some definitions in the yaml file:
DatasetType = TypedDict('DatasetType', {'path': str, 'random_seed': float, 'file_delimiter': str, 'skip_head': bool})
FilterUsersType = TypedDict('FilterUsersType', {'min_consumption': int, 'num_users': int})
FilterItemsType = TypedDict('FilterItemsType', {'min_ratings': int, 'num_items': int})
FilteringType = TypedDict('FilteringType', {'filter_users': FilterUsersType, 'filter_items': FilterItemsType})
SplittingType = TypedDict('SplittingType', {'strategy': str, 'train_size': float, 'test_consumes': int})
Validation = TypedDict('Validation', {'validation_size': float})


class DefaultLoader:

    def __init__(self,
                 dataset: DatasetType,
                 prefiltering: FilteringType,
                 splitting: SplittingType,
                 validation: Validation) -> None:
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
            # TODO: raise an error
            # raise errors.EvaluationRunNotFoundError("Could not find evaluation run")
            print("You must define your dataset path to be reader by the system.")

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
        """
        Returns:
            The data read according to the parameters specified.
        """
        data = np.loadtxt(self.dataset_path,
                          delimiter=self.delimiter,
                          skiprows=self.skip_rows)
        # TODO: implement way to define the columns (user-id, item-id, etc)
        return data

    def _filter(self,
                data: np.array) -> np.ndarray:
        """
        Args:
            data: the array of data previously read
        Returns:
            The data filtered by the filters applied.
        """
        data_df = pd.DataFrame(data)
        print(f"\nApplying filters...")
        for key, filters in self.prefiltering.items():
            print(f"{key}:")
            for filter_method, value in filters.items():
                print(f"\t {filter_method}: {value}")
                data_df = getattr(FilterRegistry.get(key), filter_method)(data_df, value)

    
        return data_df.to_numpy()

    def _split(self,
               dataset: Dataset) -> [Dataset, Dataset]:
        """
        Args:
            dataset (Dataset): an object of the dataset class
        Returns:
            train_dataset (Dataset):
            test_dataset (Dataset):
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

    def process(self) -> [Dataset, Dataset]:
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
        if len(self.prefiltering) > 0:
            filtered_data = self._filter(dataset.data)
            # update dataset
            dataset = Dataset(filtered_data)
            dataset.reset_index()
            dataset.set_parameters()
            dataset.update_num_total_users_items()

        # Apply the split approach
        print(f"\nApplying splitting strategy: {self.strategy}\n")
        train_dataset, test_dataset = self._split(dataset)
        
        # print("train:", train_dataset.num_total_items, train_dataset.num_total_users)
        # print("teste:", test_dataset.num_total_items, test_dataset.num_total_users)

        return TrainTestDataset(train_dataset, test_dataset)
