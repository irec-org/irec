from .Interactor import Interactor
class ICF(Interactor):
    def __init__(self, var, user_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = var
        self.user_lambda = user_lambda
