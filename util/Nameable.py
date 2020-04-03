from util import dict_to_list, dict_to_str

class Nameable():
    def get_name(self,parameters=None,name=None):
        if parameters == None:
            parameters = self.__dict__
        filtered_dict = {k: v for k, v in parameters.items() if isinstance(v, (int, float, complex, str))}
        list_parameters=list(map(str,dict_to_list(filtered_dict)))
        has_parameter = "_" if len(list_parameters)>0 else ""

        if name is None:
            return f"{self.__class__.__name__}"+\
                has_parameter+'_'.join(list_parameters)
        else:
            return f"{name}"+\
                has_parameter+'_'.join(list_parameters)

    def get_verbose_name(self,parameters=None,name=None):
        if parameters == None:
            parameters = self.__dict__
        filtered_dict = {k: v for k, v in parameters.items() if isinstance(v, (int, float, complex, str))}

        if name is None:
            string =  f'{self.__class__.__name__}\n'
        else:
            string =  f'{name}\n'

        string += dict_to_str(filtered_dict)
        return string
        
