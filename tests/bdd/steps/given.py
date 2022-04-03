from behave import given, step
from behave_pandas import table_to_dataframe


@given('a dataset containing')
def step_impl(context):
    context.input = context.table
    context.input_df = table_to_dataframe(context.input)


@step('using the minimum of expected ratings as "{min_ratings}"')
def step_impl(context, min_ratings):
    context.min_ratings = int(min_ratings)


@step('using the number of items as "{num_items}"')
def step_impl(context, num_items):
    context.num_items = int(num_items)


@step('using the number of users as "{num_users}"')
def step_impl(context, num_users):
    context.num_users = int(num_users)


@step('random seed equals to "{random_seed}"')
def step_impl(context, random_seed):
    context.random_seed = int(random_seed)


@step('using the train size as "{train_size}"')
def step_impl(context, train_size):
    context.train_size = float(train_size)


@step('using the tests consumes as "{test_consumes}"')
def step_impl(context, test_consumes):
    context.test_consumes = int(test_consumes)
