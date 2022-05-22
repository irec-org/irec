import numpy as np

from behave import then, step
from behave_pandas import table_to_dataframe


@then('the output should be equal to')
def step_impl(context):
    context.expected_df = table_to_dataframe(context.table)
    assert np.array_equal(context.result, context.expected_df), f"The result actually is: \n {context.result}"


@then('the array output should be equal to')
def step_impl(context):
    context.expected_df = table_to_dataframe(context.table)
    context.expected_array = np.array(context.expected_df["userId"])
    assert np.array_equal(context.result, context.expected_array), f"The result actually is: \n {context.result}"


@then("the train output should be equal to")
def step_impl(context):
    context.expected_train_df = table_to_dataframe(context.table)
    assert np.array_equal(context.train_df, context.expected_train_df), f"The result actually is: \n {context.train_df}"


@step("the test output should be equal to")
def step_impl(context):
    context.expected_test_df = table_to_dataframe(context.table)
    assert np.array_equal(context.test_df, context.expected_test_df), f"The result actually is: \n {context.test_df}"


@step("the total users should be equal to {total:d}")
def step_impl(context, total):
    testing_users = np.unique(context.test_df["userId"])
    training_users = np.unique(context.train_df["userId"])
    context.total = len(testing_users) + len(training_users)
    assert context.total == total, f"The result actually is: {context.total}\n{context.train_df}\n{context.test_df}"
