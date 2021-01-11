from . import util
class Parameterizable:
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = []
    def get_id(self):
        return self.__class__.__name__+':{'+util.dict_to_str(
            {i: getattr(self,i)
             if isinstance(getattr(self,i),Parameterizable) else
             i: getattr(self,i).get_id()
             for i in self.parameters}
        )+'}'
