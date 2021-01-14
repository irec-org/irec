from .ExperimentalInteractor import ExperimentalInteractor
class MFInteractor(ExperimentalInteractor):
    def __init__(self, num_lat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_lat = num_lat
        self.parameters.extend(['num_lat'])

