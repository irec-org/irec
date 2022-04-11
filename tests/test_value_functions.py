from irec.agents.value_functions.experimental.most_popular import MostPopular
from irec.agents.value_functions.experimental.ucb import UCB
from irec.agents.value_functions.experimental.random import Random
from irec.agents.value_functions.base import ValueFunction


def test_create_value_functions():
    assert isinstance(MostPopular(), ValueFunction)
    assert isinstance(UCB(0.1), ValueFunction)
    assert isinstance(Random(), ValueFunction)
