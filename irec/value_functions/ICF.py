from irec.value_function.ExperimentalValueFunction import ExperimentalValueFunction
from irec.value_function.MFValueFunction import MFValueFunction


class ICF(MFValueFunction):
    def __init__(self, var, user_var, item_var, stop_criteria, iterations,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = var
        self.user_var = user_var
        self.item_var = item_var
        self.stop_criteria = stop_criteria
        self.iterations = iterations

    def get_user_lambda(self):
        return self.var / self.user_var
