from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir)
from ..base import ValueFunction


class ExperimentalValueFunction(ValueFunction):
    def __init__(self, *args, **kwargs):
        ValueFunction.__init__(self, *args, **kwargs)

    def reset(self, observation):
        train_dataset = observation
        ValueFunction.reset(self, train_dataset)
