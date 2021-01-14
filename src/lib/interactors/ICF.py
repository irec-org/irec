from .ExperimentalInteractor import ExperimentalInteractor
class ICF(ExperimentalInteractor):
    def __init__(self, var, user_var, item_var, stop_criteria, iterations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = var
        self.user_var = user_var
        self.item_var = item_var
        self.stop_criteria = stop_criteria
        self.iterations = iterations
        self.parameters.extend(['var','user_var','item_var','stop_criteria','iterations'])

    def get_user_lambda(self):
        return self.var/self.user_var
