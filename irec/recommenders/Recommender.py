from collections import defaultdict


class Recommender:
    """Recommender."""

    def __init__(self, result_list_size=10, *args, **kwargs):
        """__init__.

        Args:
            result_list_size:
            args:
            kwargs:
        """
        super().__init__(*args, **kwargs)
        self.result_list_size = result_list_size
        self.results = defaultdict(list)

    def train(self):
        """train."""
        ṕass

    def predict(self):
        """predict."""
        self.results.clear()
