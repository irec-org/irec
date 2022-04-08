from os.path import dirname, realpath, sep, pardir
from sklearn.model_selection import ParameterGrid
import sklearn.model_selection
from numpy import linspace
from tqdm import tqdm
from copy import copy
import numpy as np
import itertools
import argparse
import yaml
import copy
import sys
import os

sys.path.append(dirname(dirname(realpath(__file__))))
from irec.app.utils import load_settings, flatten_dict, unflatten_dict

def generate_values(values):
	if "linspace" in str(values):
		return np.around(eval(values),3).tolist()
	return values

def replace(obj, key, val):
    return {k: replace(val if k == key else v, key, val) 
        for k,v in obj.items()} if isinstance(obj, dict) else obj

def generate_grid_parameters(template):
    d = flatten_dict(template)
    d = {k: (v if isinstance(v, (list, np.ndarray)) else [v]) for k, v in d.items()}
    t1 = list(sklearn.model_selection.ParameterGrid(d))
    return list(map(lambda x: unflatten_dict(copy.deepcopy(x)), t1))

parser = argparse.ArgumentParser()
parser.add_argument('--agents', required=True, nargs="*", type=str, help='Name of agent in agents.yaml')
args = parser.parse_args()

agents_variables = yaml.load(open("./settings/agents_variables.yaml"), Loader=yaml.SafeLoader)
agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)
settings = load_settings(dirname(realpath(__file__)))
print()

for agent_name in args.agents:

	base_agent_parameters = settings["agents"][agent_name]
	template = copy.deepcopy(base_agent_parameters)
	variables = agents_variables[agent_name]

	for var, values in variables.items():
		template = replace(template, var, generate_values(values))

	new_parameters = generate_grid_parameters(template)
	agents_search[agent_name] = new_parameters
	print(f"({len(new_parameters)}) New parameters added in {agent_name}!")

yaml.dump(agents_search, open("./settings/agents_search.yaml", "w"))
print(f"\nParameters saved in irec/app/settings/agents_search.yaml")