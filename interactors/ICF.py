class ICF(Interactor):
    def __init__(self, var, u_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.var = var
        self.u_lambda = u_lambda
