import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import yaml
d = yaml.load(open('./settings/interactors_search_parameters.yaml').read(),
              Loader=yaml.FullLoader)

import copy
new_d = copy.copy(d)
for k1, v1 in d.items():
    # print(k1,v1)
    new_d[k1] = []
    for parameters in v1:
        new_d[k1].append({
            'SimpleAgent': {
                'value_function': {
                    k1: parameters
                },
                'action_selection_policy': {
                    'Greedy': {}
                }
            }
        })
print(yaml.dump(new_d))
