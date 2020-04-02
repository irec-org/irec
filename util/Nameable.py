from util import dict_to_list

class Nameable():
    def get_name(self):
        filtered_dict = {k: v for k, v in self.__dict__.items() if isinstance(v, (int, float, complex, str))}
        list_parameters=list(map(str,dict_to_list(filtered_dict)))
        string="_" if len(list_parameters)>0 else ""
        return f"{self.__class__.__name__}"+\
            string+'_'.join(list_parameters)
