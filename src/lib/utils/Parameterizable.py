class Parameterizable:
    def __init__(self,parameters=dict(),*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters
    def get_name(self):
        if hasattr(self,'parameters'):
            return self.__class__.__name__+'_'+util.dict_to_str(self.parameters)
        else:
            raise TypeError
