from os.path import dirname, realpath, sep, pardir
from collections import defaultdict
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir )
import collections
import traceback
import matplotlib.ticker as mtick
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, realpath, sep, pardir
from lib.utils.PersistentDataManager import PersistentDataManager
from lib.utils.InteractorCache import InteractorCache
import lib.action_selection_policies
import lib.agents
import lib.value_functions
import copy
LATEX_TABLE_FOOTER = r"""
\end{tabular}
\end{document}
"""
LATEX_TABLE_HEADER = r"""
\documentclass{standalone}
%%\usepackage[landscape, paperwidth=15cm, paperheight=30cm, margin=0mm]{geometry}
\usepackage{multirow}
\usepackage{color, colortbl}
\usepackage{xcolor, soul}
\usepackage{amssymb}
\definecolor{Gray}{gray}{0.9}
\definecolor{StrongGray}{gray}{0.7}
\begin{document}
\begin{tabular}{%s}
\hline
\rowcolor{StrongGray}
Dataset & %s \\"""
LATEX_TABLE_METRICS_INTERACTIONS_HEADER = r"""
\hline
\hline
\rowcolor{Gray}
Measure & %s \\
\hline
\rowcolor{Gray}
T & %s \\
\hline
\hline
"""
import yaml

def generate_table_spec(nums_interactions_to_show, num_datasets_preprocessors):
    res = '|'
    for i in range(1 +
                   len(nums_interactions_to_show) * num_datasets_preprocessors):
        res += 'c'
        if i % (len(nums_interactions_to_show)) == 0:
            res += '|'
    return res


def generate_datasets_line(nums_interactions_to_show, preprocessors_names):
    return ' & '.join([
        r"\multicolumn{%d}{c|}{%s}" % (len(nums_interactions_to_show), i)
        for i in preprocessors_names
    ])


def generate_metrics_header_line(nums_interactions_to_show, num_preprocessors,
                                 metric_name):
    return ' & '.join(
        map(
            lambda x: r"\multicolumn{%d}{c|}{%s}" %
            (len(nums_interactions_to_show), x),
            [metric_name] * num_preprocessors))


def generate_interactions_header_line(nums_interactions_to_show,
                                      num_preprocessors):
    return ' & '.join([' & '.join(map(str, nums_interactions_to_show))] *
                      num_preprocessors)


def generate_metric_interactions_header(nums_interactions_to_show,
                                        num_preprocessors, metric_name):
    btex = LATEX_TABLE_METRICS_INTERACTIONS_HEADER % (
        generate_metrics_header_line(nums_interactions_to_show,
                                     num_preprocessors, metric_name),
        generate_interactions_header_line(nums_interactions_to_show,
                                          num_preprocessors))
    return btex

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
def defaultify(d):
    if not isinstance(d, dict):
        return d
    return defaultdict(lambda: dict, {k: defaultify(v) for k, v in d.items()})
def load_settings():
    d = dict()
    loader = yaml.SafeLoader
    d['agents_preprocessor_parameters'] = yaml.load(
        open(dirname(realpath(__file__)) + sep +"settings" + sep + "agents_preprocessor_parameters.yaml"),
        Loader=loader)
    d['agents_preprocessor_parameters'] =defaultify(d['agents_preprocessor_parameters'])

    d['interactors_general_settings'] = yaml.load(
        open(dirname(realpath(__file__)) + sep +"settings" + sep + "interactors_general_settings.yaml"),
        Loader=loader)

    d['interactors_search_parameters'] = yaml.load(
        open(dirname(realpath(__file__)) + sep +"settings" + sep + "interactors_search_parameters.yaml"),
        Loader=loader)

    d['evaluation_policies_parameters'] = yaml.load(
        open(dirname(realpath(__file__)) + sep +"settings" + sep + "evaluation_policies_parameters.yaml"),
        Loader=loader)

    with open(dirname(realpath(__file__)) + sep +"settings"+sep+"datasets_preprocessors_parameters.yaml") as f:
        d['datasets_preprocessors_parameters'] = yaml.load(f,Loader=loader)
        d['datasets_preprocessors_parameters'] = {k: {**setting, **{'name':k}}
                                  for k, setting in d['datasets_preprocessors_parameters'].items()}
    with open(dirname(realpath(__file__)) + sep +"settings"+sep+"defaults.yaml") as f:
        d['defaults'] = yaml.load(f,Loader=loader)
    return d

def load_settings_to_parser(settings,parser):
    settings_flatten=flatten_dict(settings)
    for k,v in settings_flatten.items():
        parser.add_argument(f'--{k}',default=v)
        # parser.add_argument(f'--{k}')

def sync_settings_from_args(settings,args, sep='.'):
    settings = copy.deepcopy(settings)
    args_dict = vars(args)
    settings_flatten=flatten_dict(settings)
    for i in set(args_dict.keys()).intersection(set(settings_flatten.keys())):
        tmp = settings
        for j in i.split(sep)[:-1]:
            tmp = tmp[j]
        tmp[i.split(sep)[-1]] = args_dict[i]
    return settings


def plot_similar_items(ys,method1,method2,title=None):
    fig, ax = plt.subplots()
    ax.plot(np.sort(ys)[::-1],linewidth=2,color='k')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.set_title()
    ax.set_title(title)
    ax.set_xlabel('Users Rank')
    ax.set_ylabel('Similar items (%)')
    return fig


def run_interactor(itr,evaluation_policy,dm,forced_run):
    pdm = PersistentDataManager(directory='results')
    if forced_run or not pdm.file_exists(InteractorCache().get_id(
            dm, evaluation_policy, itr)):
        try:
            history_items_recommended = evaluation_policy.evaluate(
                itr, dm.dataset_preprocessed[0],
                dm.dataset_preprocessed[1])
        except:
            print(traceback.print_exc())
            raise SystemError

        pdm = PersistentDataManager(directory='results')
        pdm.save(InteractorCache().get_id(dm, evaluation_policy, itr),
                 history_items_recommended)
    else:
        print("Already executed",
              InteractorCache().get_id(dm, evaluation_policy, itr))

def get_agent_id(agent,dataset_preprocessor_name,settings):
    pass

def create_agent(agent_name,dataset_preprocessor_name,settings):
    # try:
    agent_settings = settings['agents_preprocessor_parameters'][dataset_preprocessor_name][agent_name]
    agent_class_name = list(agent_settings.keys())[0]
    agent_parameters = list(agent_settings.values())[0]
    agent_class= eval('lib.agents.'+agent_class_name)
    action_selection_policy_name = list(agent_parameters['action_selection_policy'].keys())[0]
    action_selection_policy_parameters = list(agent_parameters['action_selection_policy'].values())[0]
    value_function_name = list(agent_parameters['value_function'].keys())[0]
    value_function_parameters = list(agent_parameters['value_function'].values())[0]

    action_selection_policy = eval('lib.action_selection_policies.'+action_selection_policy_name)(**action_selection_policy_parameters)
    value_function = eval('lib.value_functions.'+value_function_name)(**value_function_parameters)

    return agent_class(action_selection_policy=action_selection_policy,value_function=value_function,name=agent_name)
    # except:
        # print("Error creating agent")
        # raise SystemError
        # return None

def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d
