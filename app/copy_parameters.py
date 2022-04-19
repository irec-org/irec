from irec.connector import utils
import yaml
from os.path import dirname, realpath
import argparse

parser = argparse.ArgumentParser()
# argparser.add_argument('',nargs=*)
parser.add_argument('-m', nargs='*')
parser.add_argument('-sb', nargs='*')
parser.add_argument('-tb', nargs='*')
args = parser.parse_args()

settings = utils.load_settings(dirname(realpath(__file__)))
dataset_agents = utils.defaultify(yaml.load(
    open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader
))
print(args)
print(dataset_agents)
for agent_name in args.m:
    for source_base, target_base in zip(args.sb, args.tb):
        dataset_agents[target_base][
            agent_name] = dataset_agents[
                source_base][agent_name]

open('settings/dataset_agents.yaml',
     'w').write(yaml.dump(utils.default_to_regular(dataset_agents)))
