from .ExperimentalInteractor import ExperimentalInteractor
class ICF(ExperimentalInteractor):
    def __init__(self, var, user_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = var
        self.user_lambda = user_lambda
        self.parameters.extend(['var','user_lambda'])
