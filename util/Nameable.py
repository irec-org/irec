from util import dict_to_list, dict_to_str

class Nameable():
    def __init__(self,prefix_name=None):
        self.prefix_name = prefix_name

    # def __init__(self,prefix_name=None,name=None,parameters=None):
    #     self.prefix_name = prefix_name
    #     self.name = name
    #     self.parameters = parameters

    @property
    def prefix_name(self):
        return self._prefix_name[0]

    @prefix_name.setter
    def prefix_name(self,value):
        self._prefix_name = [value]

    # @property
    # def name(self):
    #     return self._name[0]

    # @name.setter
    # def name(self,value):
    #     self._name = [value]

    # @property
    # def parameters(self):
    #     return self._parameters[0]

    # @parameters.setter
    # def parameters(self,value):
    #     self._parameters = [value]

    def filter_parameters(self,parameters):
        return {k: v for k, v in parameters.items() if isinstance(v, (int, float, complex, str))}

    def get_name(self,parameters=None,name=None):
        if parameters == None:
            parameters = self.__dict__

        filtered_dict = self.filter_parameters(parameters)
        list_parameters=list(map(str,dict_to_list(filtered_dict)))

        return (self.prefix_name+'_' if self.prefix_name else '')+\
            (self.__class__.__name__ if name is None else name)+\
            ('_' if len(list_parameters)>0 else '')+\
            '_'.join(list_parameters)

    def get_verbose_name(self,parameters=None,name=None):
        if parameters == None:
            parameters = self.__dict__

        filtered_dict = self.filter_parameters(parameters)
        string = (self.prefix_name+' ' if self.prefix_name else '')+\
            (self.__class__.__name__ if name is None else name)+'\n'

        string += dict_to_str(filtered_dict)
        return string
