import random

class FilteringByUsers:

    @staticmethod
    def min_consumption(df_dataset, min_consumption):
        selected_users = dict(df_dataset.groupby(0)[1].agg("count")[lambda consumption: consumption >= min_consumption])
        return df_dataset[df_dataset[0].isin(selected_users)]
     
    @staticmethod
    def num_users(df_dataset, num_users):
        try: selected_users = random.sample(list(df_dataset[0].unique()), num_users)
        except: return df_dataset
        return df_dataset[df_dataset[0].isin(selected_users)]

class FilteringByItems:

    @staticmethod
    def min_ratings(df_dataset, min_ratings):
        selected_items = dict(df_dataset.groupby(1)[0].agg("count")[lambda ratings: ratings >= min_ratings])
        return df_dataset[df_dataset[1].isin(selected_items)]

    @staticmethod
    def num_items(df_dataset, num_items):
        try: selected_items = random.sample(list(df_dataset[1].unique()), num_items)
        except: return df_dataset
        return df_dataset[df_dataset[1].isin(selected_items)]