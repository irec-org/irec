import random
from pandas import DataFrame


class FilteringByItems:
    """FilteringByItems.
    This class contains different filtering by item approaches.
    """

    def __init__(self):
        pass

    @staticmethod
    def min_ratings(df_dataset: DataFrame, min_ratings: int) -> DataFrame:
        """min_ratings.
        This function removes items whose total number of
        ratings is less than [min_ratings].

        Args:
            df_dataset (DataFrame): the data to be filtered.
            min_ratings (int): minimum number of ratings.
        Returns:
            The data filtered by the number of ratings.
        """
        selected_items = dict(
            df_dataset.groupby("itemId")["userId"].agg("count")[
                lambda ratings: ratings >= min_ratings
            ]
        )
        return df_dataset[df_dataset["itemId"].isin(selected_items)]

    @staticmethod
    def num_items(df_dataset: DataFrame, num_items: int) -> DataFrame:
        """num_items.
        This function limits the number of distinct items in the dataset.

        Args:
            df_dataset (DataFrame): the data to be filtered.
            num_items (int): maximum number of items in the dataset.
        Returns:
            The data filtered by the number of items.
        """
        try:
            selected_items = random.sample(list(df_dataset["itemId"].unique()), num_items)
        except:
            return df_dataset
        return df_dataset[df_dataset["itemId"].isin(selected_items)]
