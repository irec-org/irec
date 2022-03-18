import random


class FilteringByUsers:

    def __init__(self):
        pass

    @staticmethod
    def min_consumption(df_dataset, min_consumption):
        selected_users = dict(
            df_dataset.groupby(0)[1].agg("count")[
                lambda consumption: consumption >= min_consumption
            ]
        )
        return df_dataset[df_dataset[0].isin(selected_users)]

    @staticmethod
    def num_users(df_dataset, num_users):
        try:
            selected_users = random.sample(list(df_dataset[0].unique()), num_users)
        except:
            return df_dataset
        return df_dataset[df_dataset[0].isin(selected_users)]
