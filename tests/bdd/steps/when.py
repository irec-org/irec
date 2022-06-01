import random

import numpy as np
import pandas as pd
from behave import when

from irec.environment.dataset import Dataset
from irec.environment.filter.filtering_by_items import FilteringByItems
from irec.environment.filter.filtering_by_users import FilteringByUsers
from irec.environment.split.randomised import Random
from irec.environment.split.temporal import Temporal
from irec.environment.split.global_timestamp import GlobalTimestampSplit
from irec.environment.split.user_history import UserHistory


@when('filtered by items with min_ratings')
def step_impl(context):
    context.result = FilteringByItems.min_ratings(context.input_df, context.min_ratings)


@when('filtered by items with num_items')
def step_impl(context):
    random.seed(context.random_seed)
    context.result = FilteringByItems.num_items(context.input_df, context.num_items)


@when('filtered by users with min_consumption')
def step_impl(context):
    context.result = FilteringByUsers.min_consumption(context.input_df, context.min_ratings)


@when('filtered by users with num_users')
def step_impl(context):
    context.result = FilteringByUsers.num_users(context.input_df, context.num_users)


@when('split randomly')
def step_impl(context):
    # parameters
    num_users = len(context.input_df["userId"].unique())
    num_train_users = round(num_users * context.train_size)
    num_test_users = int(num_users - num_train_users)
    dtypes = {"userId": int,
              "itemId": int,
              "rating": float,
              "timestamp": int}
    # dataset
    dataset = Dataset(data=context.input_df.to_numpy())
    dataset.reset_index()
    # run metric
    r = Random(train_size=context.train_size,
               test_consumes=context.test_consumes)
    test_uids = r.get_test_uids(dataset.data, num_test_users)
    train_dataset, test_dataset = r.split_dataset(dataset.data, test_uids)
    # train
    context.train_df = pd.DataFrame(train_dataset.data, columns=dtypes.keys())
    context.train_df = context.train_df.astype(dtypes)
    # test
    context.test_df = pd.DataFrame(test_dataset.data, columns=dtypes.keys())
    context.test_df = context.test_df.astype(dtypes)


@when('split user_history')
def step_impl(context):
    # parameters
    num_users = len(context.input_df["userId"].unique())
    num_train_users = round(num_users * context.train_size)
    num_test_users = int(num_users - num_train_users)
    dtypes = {"userId": int,
              "itemId": int,
              "rating": float,
              "timestamp": int}
    # dataset
    dataset = Dataset(data=context.input_df.to_numpy())
    dataset.reset_index()
    # run metric
    t = UserHistory(train_size=context.train_size,
                 test_consumes=context.test_consumes)
    test_uids = t.get_test_uids(dataset.data, num_test_users)
    train_dataset, test_dataset = t.split_dataset(dataset.data, test_uids)
    # train
    context.train_df = pd.DataFrame(train_dataset.data, columns=dtypes.keys())
    context.train_df = context.train_df.astype(dtypes)
    # test
    context.test_df = pd.DataFrame(test_dataset.data, columns=dtypes.keys())
    context.test_df = context.test_df.astype(dtypes)

@when('split temporal')
def step_impl(context):
    # parameters
    num_users = len(context.input_df["userId"].unique())
    num_train_users = round(num_users * context.train_size)
    num_test_users = int(num_users - num_train_users)
    dtypes = {"userId": int,
              "itemId": int,
              "rating": float,
              "timestamp": int}
    # dataset
    dataset = Dataset(data=context.input_df.to_numpy())
    dataset.reset_index()
    # run metric
    t = Temporal(train_size=context.train_size,
                 test_consumes=context.test_consumes)
    test_uids = t.get_test_uids(dataset.data, num_test_users)
    train_dataset, test_dataset = t.split_dataset(dataset.data, test_uids)
    # train
    context.train_df = pd.DataFrame(train_dataset.data, columns=dtypes.keys())
    context.train_df = context.train_df.astype(dtypes)
    # test
    context.test_df = pd.DataFrame(test_dataset.data, columns=dtypes.keys())
    context.test_df = context.test_df.astype(dtypes)


@when('split globally')
def step_impl(context):
    # parameters
    num_users = len(context.input_df["userId"].unique())
    num_train_users = round(num_users * context.train_size)
    num_test_users = int(num_users - num_train_users)
    dtypes = {
        "userId": int,
        "itemId": int,
        "rating": float,
        "timestamp": int
    }
    # dataset
    dataset = Dataset(data=context.input_df.to_numpy())
    dataset.reset_index()
    # run metric
    t = GlobalTimestampSplit(train_size=context.train_size,
                             test_consumes=context.test_consumes)
    test_uids = t.get_test_uids(dataset.data, num_test_users)
    train_dataset, test_dataset = t.split_dataset(dataset.data, test_uids)
    # train
    context.train_df = pd.DataFrame(train_dataset.data, columns=dtypes.keys())
    context.train_df = context.train_df.astype(dtypes)
    # test
    context.test_df = pd.DataFrame(test_dataset.data, columns=dtypes.keys())
    context.test_df = context.test_df.astype(dtypes)
