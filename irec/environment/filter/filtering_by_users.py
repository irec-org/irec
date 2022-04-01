import random
from pandas import DataFrame


class FilteringByUsers:
    """FilteringByUsers.
    This class contains different filtering by users approaches.
    """

    def __init__(self):
        pass

    @staticmethod
    def min_consumption(df_dataset: DataFrame, min_consumption: int) -> DataFrame:
        """min_consumption.
        This function removes users whose total number of
        consumptions is less than [min_consumption].

        Args:
            df_dataset (DataFrame): the data to be filtered.
            min_consumption (int): minimum number of items consumed by a user.
        Returns:
            The data filtered by the number of consumptions.
        """
        selected_users = dict(
            df_dataset.groupby(0)[1].agg("count")[
                lambda consumption: consumption >= min_consumption
            ]
        )
        return df_dataset[df_dataset[0].isin(selected_users)]

    @staticmethod
    def num_users(df_dataset: DataFrame, num_users: int) -> DataFrame:
        """num_users.
        This function limits the number of distinct users in the dataset.

        Args:
            df_dataset (DataFrame): the data to be filtered.
            num_users (int): maximum number of users in the dataset.
        Returns:
            The data filtered by the number of users.
        """
        try:
            selected_users = random.sample(list(df_dataset[0].unique()), num_users)
        except:
            return df_dataset
        return df_dataset[df_dataset[0].isin(selected_users)]
