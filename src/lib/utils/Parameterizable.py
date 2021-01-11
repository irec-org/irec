class Parameterizable:
    def __init__(self,parameters=[],*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = parameters
    def get_name(self):
        return self.__class__.__name__+'_'+util.dict_to_str(
            {i: getattr(self,i)
             for i in self.parameters}
        )
