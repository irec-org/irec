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
parser.add_argument("--agents", nargs="*", default=settings["defaults"]["agent"])
args = parser.parse_args()
agents_search = yaml.load(open("./settings/agents_search.yaml"), Loader=yaml.SafeLoader)

for agent_name in args.agents:
    base_agent_parameters = settings["agents"][agent_name]
    template = copy.deepcopy(base_agent_parameters)
    if agent_name == "PTS":
        template["SimpleAgent"]["value_function"]["PTS"]["num_lat"] = [
            5,
            10,
            20,
            50,
            100,
        ]
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

    elif agent_name == "ICTRTS":
        template["SimpleAgent"]["value_function"]["ICTRTS"]["num_lat"] = [
            5,
            10,
            20,
            50,
            100,
        ]
        template["SimpleAgent"]["value_function"]["ICTRTS"]["num_particles"] = [
            2,
            5,
            10,
            20,
            30,
        ]
    elif agent_name == "NICF":
        template["SimpleAgent"]["value_function"]["NICF"]["latent_factor"] = [
            5,
            10,
            20,
            50,
            100,
        ]
        template["SimpleAgent"]["value_function"]["NICF"]["batch"] = [
            64,
            128,
            256,
        ]
        template["SimpleAgent"]["value_function"]["NICF"]["num_blocks"] = [1, 2, 3]

        template["SimpleAgent"]["value_function"]["NICF"]["num_heads"] = [1, 2, 3]
        template["SimpleAgent"]["value_function"]["NICF"]["rnn_layer"] = [1, 2]
        template["SimpleAgent"]["value_function"]["NICF"]["dropout_rate"] = [
            0.1,
            0.2,
            0.3,
        ]
    else:

        raise IndexError("Unrecognized agent")

    # new_parameters = []
    # new_parameters.extend(generate_grid_parameters(template))
    new_parameters = generate_grid_parameters(template)
    agents_search[agent_name] = new_parameters

yaml.dump(agents_search, open("./settings/agents_search.yaml", "w"))
