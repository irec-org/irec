from irec.recommendation.agents.value_functions.most_popular import MostPopular
from irec.recommendation.agents.value_functions.ucb import UCB
from irec.recommendation.agents.value_functions.random import Random
from irec.recommendation.agents.value_functions.base import ValueFunction
from irec.recommendation.agents.value_functions.cluster_bandit import ClusterBandit


def test_create_value_functions():
    assert isinstance(MostPopular(), ValueFunction)
    assert isinstance(UCB(0.1), ValueFunction)
    assert isinstance(Random(), ValueFunction)
    assert isinstance(ClusterBandit(8,1,1,5,10), ValueFunction)
