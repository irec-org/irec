import utils
import yaml
import copy
import argparse

parser = argparse.ArgumentParser()
# argparser.add_argument('',nargs=*)
parser.add_argument('-m', nargs='*')
parser.add_argument('-sb', nargs='*')
parser.add_argument('-tb', nargs='*')
args = parser.parse_args()

settings = utils.load_settings()
print(args)
for agent_name in args.m:
    for source_base, target_base in zip(args.sb,args.tb):
        settings['agents_preprocessor_parameters'][target_base][agent_name] = settings['agents_preprocessor_parameters'][source_base][agent_name]

open('settings/agents_preprocessor_parameters.yaml',
     'w').write(yaml.dump(settings['agents_preprocessor_parameters']))
