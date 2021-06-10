import utils
import yaml
import copy
import argparse

parser = argparse.ArgumentParser()
# argparser.add_argument('',nargs=*)
parser.add_argument('-m', nargs='*')
parser.add_argument('-b', nargs='*')
args = parser.parse_args()

settings = utils.load_settings()

for agent_name in args.m:
    new_agent_name = 'Reranked' + agent_name
    search_parameters = []
    for base_name in args.b:
        agent_parameters = settings['agents_preprocessor_parameters'][
            base_name][agent_name]
        action_selection_policies = [
            'ASPReranker: {rule: {Entropy: {}},input_filter_size: 10,rerank_limit: 2}',
            'ASPReranker: {rule: {Entropy: {}},input_filter_size: 20,rerank_limit: 2}',
            'ASPReranker: {rule: {Entropy: {}},input_filter_size: 10,rerank_limit: 4}',
            'ASPReranker: {rule: {Entropy: {}},input_filter_size: 20,rerank_limit: 4}',
            'ASPReranker: {rule: {LogPopEnt: {}},input_filter_size: 10,rerank_limit: 2}',
            'ASPReranker: {rule: {LogPopEnt: {}},input_filter_size: 20,rerank_limit: 2}',
            'ASPReranker: {rule: {LogPopEnt: {}},input_filter_size: 10,rerank_limit: 4}',
            'ASPReranker: {rule: {LogPopEnt: {}},input_filter_size: 20,rerank_limit: 4}',
            'ASPReranker: {rule: {Random: {}},input_filter_size: 10,rerank_limit: 2}',
            'ASPReranker: {rule: {Random: {}},input_filter_size: 20,rerank_limit: 2}',
            'ASPReranker: {rule: {Random: {}},input_filter_size: 10,rerank_limit: 4}',
            'ASPReranker: {rule: {Random: {}},input_filter_size: 20,rerank_limit: 4}',
        ]
        for action_selection_policy in action_selection_policies:
            _agent_parameters = copy.deepcopy(agent_parameters)
            _agent_parameters['action_selection_policy'] = yaml.safe_load(action_selection_policies)
            search_parameters.append(_agent_parameters)
    settings['agents_search_parameters'][new_agent_name] = search_parameters

open('settings/agents_preprocessor_parameters.yaml','w').write(yaml.dump(
    utils.default_to_regular(settings['agents_search_parameters'])
    ))

