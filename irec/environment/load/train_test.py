from typing import TypedDict

import numpy as np

from irec.environment.dataset import Dataset

TrainDatasetType = TypedDict('TrainDatasetType', {'path': str, 'file_delimiter': str, 'skip_head': bool})
TestDatasetType = TypedDict('TestDatasetType', {'path': str, 'file_delimiter': str, 'skip_head': bool})
ValidationDatasetType = TypedDict('ValidationDatasetType', {'path': str, 'file_delimiter': str, 'skip_head': bool})
DatasetType = TypedDict('DatasetType', {'train': TrainDatasetType, 'test': TestDatasetType})


class TrainTestLoader:

    def __init__(
            self,
            dataset: DatasetType) -> None:
        """__init__.

        Args:
            dataset (DatasetType): info required by the dataset
        """
        
        assert len(dataset.keys()) == 2, "You must define files for train and test sets."
        self.dataset_params = dataset

    def _set_attributes(self,
                        dataset: DatasetType,
                        split_type: str) -> None:
        
        """_set_attributes

            Set dataset attributes

        Args:
            dataset (DatasetType): dictionary with training and test datasets
            split_type (str): split type (train or test)

        """


        if split_type in dataset.keys() and "path" in dataset[split_type].keys():
            self.path = dataset[split_type]["path"]
            self.delimiter = dataset[split_type]["file_delimiter"] \
                if "file_delimiter" in dataset[split_type].keys() else ","
            self.skip_rows = int(dataset[split_type]["skip_head"]) \
                if "skip_head" in dataset[split_type].keys() else 1
        else:
            raise IndexError(f"You must define your {split_type} data and its path to be reader by the system.")

    @staticmethod
    def _read(path: str,
              delimiter: str,
              skiprows: int) -> np.ndarray:
        """_read

            The data read according to the parameters specified.

        Args:
            path (str): dataset directory
            delimiter (str): file delimiter
            skiprows (str): used to skip or not the file header

        Return:
            data (np.ndarray): the data    
        """
        data = np.loadtxt(path,
                          delimiter=delimiter,
                          skiprows=skiprows)
        return data

    def process(self) -> [Dataset, Dataset]:
        
        """process

            reads the dataset and gets information about the dataset

        Returns:
            train_dataset (Dataset): the train
            test_dataset (Dataset): the test
        """

        self._set_attributes(self.dataset_params, split_type="train")
        train_data = self._read(self.path,
                                self.delimiter,
                                self.skip_rows)
        train_dataset = Dataset(train_data)
        train_dataset.set_parameters()

        self._set_attributes(self.dataset_params, split_type="test")
        test_data = self._read(self.path,
                               self.delimiter,
                               self.skip_rows)
        test_dataset = Dataset(test_data)
        test_dataset.set_parameters()
        
        num_total_users = max(train_dataset.max_uid, test_dataset.max_uid)+1
        num_total_items = max(train_dataset.max_iid, test_dataset.max_iid)+1
       
        train_dataset.update_num_total_users_items(
            num_total_users=num_total_users, 
            num_total_items=num_total_items
        )
        test_dataset.update_num_total_users_items(
            num_total_users=num_total_users, 
            num_total_items=num_total_items
        )
        print("Test shape:", test_dataset.data.shape)
        print("Train shape:", train_dataset.data.shape)

        return [train_dataset, test_dataset]
 