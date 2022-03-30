from utils import create_mock_dataset


class TestFilteringByItems:

    def test_num_items(self):
        l = create_mock_dataset()
        assert len(l) == 3

    def test_min_ratings(self):
        assert True
