from irec.connector.utils import flatten_dict, unflatten_dict
import sklearn.model_selection
from .base import Tunning
from numpy import linspace
# from base import Tunning
from typing import List
import numpy as np
import json
import copy

class GridSearch(Tunning):


    def _replace(self, obj, key, val):
        return {k: self._replace(val if k == key else v, key, val) 
            for k,v in obj.items()} if isinstance(obj, dict) else obj
    
    def _update_current_template(self, template):
        for key, value in template.items():        
            if isinstance(value, dict):
                self._update_current_template(value)
            else:            
                self.current_template = self._replace(self.current_template, key, self._generate_values(value))

    def _generate_values(self, values):
        if "linspace" in str(values):
            return np.around(eval(values),3).tolist()
        return values
    
    def _generate_grid_parameters(self, template):
        d = flatten_dict(template)
        d = {k: (v if isinstance(v, (list, np.ndarray)) else [v]) for k, v in d.items()}
        t1 = list(sklearn.model_selection.ParameterGrid(d))
        return list(map(lambda x: unflatten_dict(copy.deepcopy(x)), t1))


    def generate_settings(self, agents_variables):

        if not isinstance(agents_variables, List):
            agents_variables = [agents_variables]
        
        agents_search = {}
        for index, agent_template in enumerate(agents_variables):
            agent_name = list(agents_variables[index].keys())[0]
            self.current_template = agent_template[agent_name]
            self._update_current_template(self.current_template)
            agents_variables[index] = self.current_template
            # print(self.current_template)
            new_parameters = self._generate_grid_parameters(agents_variables[index])
            agents_search[agent_name] = new_parameters
        return agents_search



# g = GridSearch()

# x = {
#     "Egreedy": {
#         "SimpleAgent": {
#             "action_selection_policy": {
#                 "ASPEGreedy": {
#                     "epsilon": "linspace(0.1, 1, 5)"
#                 }
#             },
#             "value_function": {
#                 "EGreedy": {}
#             }
#         }, 
#     }
# }

# template = g.generate_settings(x)
# print (json.dumps(template, indent=2))