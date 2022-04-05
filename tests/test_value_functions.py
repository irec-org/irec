from irec.agents.value_functions.most_popular import MostPopular
from irec.agents.value_functions.ucb import UCB
from irec.agents.value_functions.random import Random
from irec.agents.value_functions.value_function import ValueFunction


def test_create_value_functions():
    assert isinstance(MostPopular(), ValueFunction)
    assert isinstance(UCB(0.1), ValueFunction)
    assert isinstance(Random(), ValueFunction)
