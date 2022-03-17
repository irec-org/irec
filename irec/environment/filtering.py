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


  

  
#    def filter_users(self, df_dataset, filters):
#         if filters == None: return df_dataset
#         def min_consumption(df_dataset, min_consumption):
#             selected_users = dict(df_dataset.groupby(0)[1].agg("count")[lambda consumption: consumption >= min_consumption])
#             return df_dataset[df_dataset[0].isin(selected_users)]

#         def num_users(df_dataset, num_users):
#             try: selected_users = random.sample(list(df_dataset[0].unique()), num_users)
#             except: return df_dataset
#             return df_dataset[df_dataset[0].isin(selected_users)]

#         for filter_user in filters:
#             df_dataset = eval(filter_user)(df_dataset, filters[filter_user])
 
#         return df_dataset

    # def filter_items(self, df_dataset, filters):

    #     def min_ratings(df_dataset, min_ratings):
    #         selected_items = dict(df_dataset.groupby(1)[0].agg("count")[lambda ratings: ratings >= min_ratings])
    #         return df_dataset[df_dataset[1].isin(selected_items)]

    #     def num_items(df_dataset, num_items):
    #         try: selected_items = random.sample(list(df_dataset[1].unique()), num_items)
    #         except: return df_dataset
    #         return df_dataset[df_dataset[1].isin(selected_items)]

    #     for filter_item in filters:
    #         df_dataset = eval(filter_item)(df_dataset, filters[filter_item])

    #     return df_dataset



    # def prefiltering(self, ds, filters):

    #     del filters["test_consumes"]
    #     if len(filters) == 0: return ds
    #     data_df = pd.DataFrame(ds.data)

    #     print("Applying filters...")
    #     for key, filters in filters.items():
    #         print("\t", key, filters)
    #         data_df = eval(f"self.{key}")(data_df, filters)

    #     dataset = Dataset(data_df.to_numpy())
    #     dataset.set_parameters()
        
    #     return dataset
