import collections
from os.path import dirname, realpath, sep, pardir
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

def load_settings():
    d = dict()
    loader = yaml.SafeLoader
    d['interactors_preprocessor_paramaters'] = yaml.load(
        open(dirname(realpath(__file__)) + sep +"settings" + sep + "interactors_preprocessor_parameters.yaml"),
        Loader=loader)

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
