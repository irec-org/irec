from typing import TypedDict

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

    def _read(self):
        pass

    def _filter(self):
        pass

    def _split(self):
        pass

    def process(self):

        self._read()
        self._filter()
        self._split()
