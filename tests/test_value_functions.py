from irec.agents.value_functions.MostPopular import MostPopular
from irec.agents.value_functions.UCB import UCB
from irec.agents.value_functions.Random import Random
from irec.agents.value_functions.ValueFunction import ValueFunction


def test_create_value_functions():
    assert isinstance(MostPopular(), ValueFunction)
    assert isinstance(UCB(0.1), ValueFunction)
    assert isinstance(Random(), ValueFunction)
