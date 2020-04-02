from util import dict_to_list, dict_to_str

class Nameable():
    def get_name(self,parameters=None):
        if parameters == None:
            parameters = self.__dict__
        filtered_dict = {k: v for k, v in parameters.items() if isinstance(v, (int, float, complex, str))}
        list_parameters=list(map(str,dict_to_list(filtered_dict)))
        string="_" if len(list_parameters)>0 else ""
        return f"{self.__class__.__name__}"+\
            string+'_'.join(list_parameters)
    def get_verbose_name(self,parameters=None):
        if parameters == None:
            parameters = self.__dict__
        filtered_dict = {k: v for k, v in parameters.items() if isinstance(v, (int, float, complex, str))}
        string =  f'{self.__class__.__name__}\n'
        string += dict_to_str(filtered_dict)
        return string
        
