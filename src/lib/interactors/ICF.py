from .ExperimentalInteractor import ExperimentalInteractor
class ICF(ExperimentalInteractor):
    def __init__(self, var, user_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters['var'] = var
        self.parameters['user_lambda'] = user_lambda
