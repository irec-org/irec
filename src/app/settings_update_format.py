import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import yaml
d = yaml.load(
    open('./settings/interactors_preprocessor_parameters.yaml').read(),
    Loader=yaml.FullLoader)

import copy
new_d = copy.copy(d)
for k1, v1 in d.items():
    for k2, v2 in v1.items():
        if v2==None:
            p = dict()
        else:
            p =v2['parameters']
        new_d[k1][k2] =  {'SimpleAgent':{
                    'value_function': {k2:p},
                'action_selection_policy': 'Greedy'
                }
        }
print(yaml.dump(new_d))
