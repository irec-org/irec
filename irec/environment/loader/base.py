from typing import Tuple, List
import numpy as np

from irec.environment.dataset import Dataset


class Loader:

    def _read(self) -> np.ndarray:
        """_read
            The data read according to the parameters specified.
            The expected columns are userId, itemId, rating, timestamp
        Returns:
            data (np.ndarray): The data loaded
        """
        pass

    def _filter(self,
                data: np.array) -> np.ndarray:
        """_filter
            Applies all filters required.
        Args:
            data: the array of data previously read
        Returns:
            data_df (np.array): The data filtered by the filters applied.
        """
        pass

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
        pass

    def process(self) -> List[Dataset, Dataset, Dataset, Dataset]:
        """process
            Perform complete processing of the dataset:
            read -> filter (optional) -> split
        Returns:
            train_dataset (Dataset): the train
            test_dataset (Dataset): the test
            x_validation (Dataset): the validation train
            y_validation (Dataset): the validation test
        """
        pass
