from util import dict_to_list, dict_to_str

class Nameable():
    def __init__(self,name_prefix=None,name_suffix=None,*args,**kwargs):
        self.name_prefix = name_prefix
        self.name_suffix = name_suffix

    def filter_parameters(self,parameters):
        return {k: v for k, v in parameters.items() if (isinstance(v, (int, float, complex, str)) and k not in ['name_suffix','name_prefix'])}

    def get_name(self,parameters=None,name=None):
        if parameters == None:
            parameters = self.__dict__

        filtered_dict = self.filter_parameters(parameters)
        list_parameters=list(map(str,dict_to_list(filtered_dict)))

        return (self.name_prefix+'_' if self.name_prefix else '')+\
            (self.__class__.__name__ if name is None else name)+\
            ('_'+self.name_suffix if self.name_suffix else '')+\
            ('_' if len(list_parameters)>0 else '')+\
            '_'.join(list_parameters)

    def get_verbose_name(self,parameters=None,name=None):
        if parameters == None:
            parameters = self.__dict__

        filtered_dict = self.filter_parameters(parameters)
        string = (self.name_prefix+' ' if self.name_prefix else '')+\
            (self.__class__.__name__ if name is None else name)+\
            (self.name_suffix if self.name_suffix else '')+'\n'

        string += dict_to_str(filtered_dict)
        return string
