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

action_selection_policies = {
    'Entropy': [
        '{ASPReranker: {rule: {Entropy: {}},input_filter_size: 10,rerank_limit: 2}}',
        '{ASPReranker: {rule: {Entropy: {}},input_filter_size: 20,rerank_limit: 2}}',
        '{ASPReranker: {rule: {Entropy: {}},input_filter_size: 10,rerank_limit: 4}}',
        '{ASPReranker: {rule: {Entropy: {}},input_filter_size: 20,rerank_limit: 4}}',
    ],
    'LogPopEnt': [
        '{ASPReranker: {rule: {LogPopEnt: {}},input_filter_size: 10,rerank_limit: 2}}',
        '{ASPReranker: {rule: {LogPopEnt: {}},input_filter_size: 20,rerank_limit: 2}}',
        '{ASPReranker: {rule: {LogPopEnt: {}},input_filter_size: 10,rerank_limit: 4}}',
        '{ASPReranker: {rule: {LogPopEnt: {}},input_filter_size: 20,rerank_limit: 4}}',
    ],
    'Random': [
        '{ASPReranker: {rule: {Random: {}},input_filter_size: 10,rerank_limit: 2}}',
        '{ASPReranker: {rule: {Random: {}},input_filter_size: 20,rerank_limit: 2}}',
        '{ASPReranker: {rule: {Random: {}},input_filter_size: 10,rerank_limit: 4}}',
        '{ASPReranker: {rule: {Random: {}},input_filter_size: 20,rerank_limit: 4}}',
    ],
    'MostPopular': [
        '{ASPReranker: {rule: {MostPopular: {}},input_filter_size: 10,rerank_limit: 2}}',
        '{ASPReranker: {rule: {MostPopular: {}},input_filter_size: 20,rerank_limit: 2}}',
        '{ASPReranker: {rule: {MostPopular: {}},input_filter_size: 10,rerank_limit: 4}}',
        '{ASPReranker: {rule: {MostPopular: {}},input_filter_size: 20,rerank_limit: 4}}',
    ],
}

for agent_name in args.m:
    for reranked_rule, _action_selection_policies in action_selection_policies.items(
    ):
        new_agent_name = 'Reranked' + reranked_rule + agent_name

        search_parameters = []
        for base_name in args.b:
            agent_parameters = settings['agents_preprocessor_parameters'][
                base_name][agent_name]
            for action_selection_policy in _action_selection_policies:
                _agent_parameters = copy.deepcopy(agent_parameters)
                # print(yaml.safe_load(action_selection_policy))
                _agent_parameters[list(_agent_parameters.keys(
                ))[0]]['action_selection_policy'] = yaml.safe_load(
                    action_selection_policy)
                # _agent_parameters['action_selection_policy'] = yaml.safe_load(
                # action_selection_policy)
                search_parameters.append(_agent_parameters)
                # print(utils.default_to_regular(_agent_parameters))
        settings['agents_search_parameters'][
            new_agent_name] = search_parameters

open('settings/agents_search_parameters.yaml',
     'w').write(yaml.dump(settings['agents_search_parameters']))
