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

        assert len(dataset.keys()) == 3, "You must define files for train and test sets."
        self.dataset_params = dataset

    def _set_attributes(self,
                        dataset: DatasetType,
                        split_type: str):

        if split_type in dataset.keys() and "path" in dataset[split_type].keys():
            self.path = dataset[split_type]["path"]
            self.delimiter = dataset[split_type]["file_delimiter"] \
                if "file_delimiter" in dataset[split_type].keys() else ","
            self.skip_rows = int(dataset[split_type]["skip_head"]) \
                if "skip_head" in dataset[split_type].keys() else 1
        else:
            # TODO: raise an error
            print(f"You must define your {split_type} data and its path to be reader by the system.")

    @staticmethod
    def _read(path: str,
              delimiter: str,
              skiprows: int) -> np.ndarray:
        """
        Returns:
            The data read according to the parameters specified.
        """
        data = np.loadtxt(path,
                          delimiter=delimiter,
                          skiprows=skiprows)
        # TODO: implement way to define the columns (user-id, item-id, etc)
        return data

    def process(self) -> [Dataset, Dataset]:

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

        return train_dataset, test_dataset
 