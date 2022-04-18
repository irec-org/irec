from ..experimental.experimental_valueFunction import ExperimentalValueFunction


class MFValueFunction(ExperimentalValueFunction):

    """MFValueFunction

    Base class used by all methods that use matrix factorization
    
    """

    def __init__(self, num_lat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_lat = num_lat

