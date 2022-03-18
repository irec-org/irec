import random

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