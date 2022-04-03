from .ExperimentalValueFunction import ExperimentalValueFunction


class MFValueFunction(ExperimentalValueFunction):
    def __init__(self, num_lat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_lat = num_lat

