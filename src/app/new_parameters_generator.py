from tqdm import tqdm
from copy import copy
import numpy as np
import itertools
import argparse
import yaml

def rrange(start, end, step):
    i = start
    while i < end:
        yield i
        i += step
    yield end

parser = argparse.ArgumentParser()
parser.add_argument('-m', required=True, type=str, help='Name of model in agents_skeletons.yaml')
args = vars(parser.parse_args())

agent = args["m"]
agent_skeletons = dict()
loader = yaml.SafeLoader
agent_skeletons = yaml.load(open("./settings/agents_skeletons.yaml"), Loader=loader)

parameters = agent_skeletons[agent]["variable_values"]
parameters = [eval(f"r{value}") if str(value).find("range") != -1 else value for var, value in parameters.items()]
parameters = [[value] if isinstance(value, int) or isinstance(value, float) or isinstance(value, bool) or isinstance(value, str) else value for value in parameters]
parameters = list(itertools.product(*parameters))

list_prmt = [dict() for i in range(len(parameters))]
{list_prmt[i].update({var:value}) for i, p in enumerate(parameters) for var, value in zip(agent_skeletons[agent]["variable_values"].keys(),p)}

skeleton = agent_skeletons[agent]["skeleton"]
skeleton_list = []

for prmt in tqdm(list_prmt, position=0, leave=True):
	sk = str(skeleton)
	for var, value in prmt.items():
		if isinstance(value, str): value = "'"+value+"'"
		sk = sk.replace(f"'{var}': None", f"'{var}': {value}")
	skeleton_list.append(eval(sk))

print("\nSkeleton_list saved in", f"{agent}_search_parameters.yaml")
file = open(f"{agent}_search_parameters.yaml", "w")
file.write(f"{agent}:\n")
yaml.dump(skeleton_list, file)
file.close()