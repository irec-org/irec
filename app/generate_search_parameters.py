#!/usr/bin/python3
from os.path import dirname, realpath, sep, pardir
import os
import sys

sys.path.append(dirname(dirname(realpath(__file__))))
from app.utils import flatten_dict, unflatten_dict
import yaml
from sklearn.model_selection import ParameterGrid
import utils
import argparse
import copy
import numpy as np
import sklearn.model_selection


def erange(start, end, step):
    return range(start, end + step, step)


def generate_grid_parameters(template):
    d = flatten_dict(template)
    d = {k: (v if isinstance(v, (list, np.ndarray)) else [v]) for k, v in d.items()}
    t1 = list(sklearn.model_selection.ParameterGrid(d))
    return list(map(unflatten_dict, t1))


settings = utils.load_settings()
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--agent", default=settings["defaults"]["agent"])
args = parser.parse_args()
agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)
base_agent_parameters = settings["agents"][args.agent]
template = copy.deepcopy(base_agent_parameters)
if args.agent == "PTS":
    # alpha = list(np.arange(0.25, 1, 0.25)) + [1, 2, 4, 8, 16, 32]
    # num_particles = [2, 5, 10, 20, 30]
    # num_lat = [5, 10, 20, 50, 100]
    template["SimpleAgent"]["value_function"]["PTS"]["num_lat"] = [5, 10, 20, 50, 100]
    template["SimpleAgent"]["value_function"]["PTS"]["num_particles"] = [
        2,
        5,
        10,
        20,
        30,
    ]
    template["SimpleAgent"]["value_function"]["PTS"]["var_v"] = np.around(
        np.linspace(0.3, 2, 6), 3
    ).tolist()
    template["SimpleAgent"]["value_function"]["PTS"]["var_u"] = np.around(
        np.linspace(0.3, 2, 6), 3
    ).tolist()
    template["SimpleAgent"]["value_function"]["PTS"]["var"] = np.around(
        np.linspace(0.3, 2, 6), 3
    ).tolist()
else:
    raise IndexError("Unrecognized agent")

print(yaml.dump(template))
new_parameters = []
new_parameters.extend(generate_grid_parameters(template))
# for a in alpha:
# for nl in num_lat:
# base_agent_parameters["SimpleAgent"]["value_function"]["PTS"]["alpha"] = a
# new_parameters += [copy.deepcopy(base_agent_parameters)]
# print(agents_search)
agents_search[args.agent] = new_parameters
print(len(agents_search))
yaml.dump(agents_search, open("./settings/agents_search.yaml", "w"))
# print(yaml.dump(new_parameters, default_flow_style=False))

# settings['agents']['LinUCB']['SimpleAgent']


# agents_parameters = ParameterGrid
