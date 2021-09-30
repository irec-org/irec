#!/usr/bin/python3
from os.path import dirname, realpath, sep, pardir
import mlflow
from mlflow.tracking import MlflowClient
import pickle
import os
import sys
import json
import utils

sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "irec")
from app import constants

from app import utils
import scipy
import copy
import value_functions
import mf
import numpy as np
import scipy.sparse
import yaml
import irec.evaluation_policies
from irec.metric_evaluators import (
    CumulativeInteractionMetricEvaluator,
    UserCumulativeInteractionMetricEvaluator,
)
from irec.utils.dataset import Dataset
import metrics
import matplotlib.pyplot as plt
from cycler import cycler
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--evaluation_policy")
parser.add_argument("--dataset_loaders", nargs="*")
parser.add_argument("--agents", nargs="*")
parser.add_argument("--metrics", nargs="*")
parser.add_argument("--metric_evaluator")
parser.add_argument("-r", type=str, default=None)
parser.add_argument("--type", default=None)
parser.add_argument("--dump", default=False, action="store_true")
settings = utils.load_settings()
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

plt.rcParams["axes.prop_cycle"] = cycler(color="krbgmyc")
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 15

dataset_agents = yaml.load(
    open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader
)

# metrics_classes = [metrics.Hits, metrics.Recall]
metrics_classes = [eval("metrics." + i) for i in args.metrics]

# metrics_classes = [
# metrics.Hits,
# metrics.Recall,
# # metrics.EPC,
# # metrics.Entropy,
# # metrics.UsersCoverage,
# # metrics.ILD,
# # metrics.GiniCoefficientInv,
# ]
metrics_classes_names = list(map(lambda x: x.__name__, metrics_classes))
metrics_names = metrics_classes_names
datasets_names = args.dataset_loaders

metric_evaluator_name = settings["defaults"]["metric_evaluator"]
metric_evaluator_parameters = settings["metric_evaluators"][metric_evaluator_name]

metric_evaluator = eval("irec.metric_evaluators." + metric_evaluator_name)(
    None, **metric_evaluator_parameters
)

evaluation_policy_name = settings["defaults"]["evaluation_policy"]
evaluation_policy_parameters = settings["evaluation_policies"][evaluation_policy_name]
evaluation_policy = eval("irec.evaluation_policies." + evaluation_policy_name)(
    **evaluation_policy_parameters
)
# metrics_names = [
# 'Cumulative Precision',
# 'Cumulative Recall',
# # 'Cumulative EPC',
# # 'Cumulative Entropy',
# # 'Cumulative Users Coverage',
# # 'Cumulative ILD',
# # '1-(Gini-Index)'
# ]
# metrics_weights = {'Entropy': 0.5,'EPC':0.5}
# metrics_weights = {'Hits': 0.3,'Recall':0.3,'EPC':0.1,'UsersCoverage':0.1,'ILD':0.1,'GiniCoefficientInv':0.1}
# metrics_weights = {'Hits': 0.3,'Recall':0.3,'EPC':0.16666,'UsersCoverage':0.16666,'ILD':0.16666}
# metrics_weights = {'Hits': 0.25,'Recall':0.25,'EPC':0.125,'UsersCoverage':0.125,'ILD':0.125,'GiniCoefficientInv':0.125}
metrics_weights = {i: 1 / len(metrics_classes_names) for i in metrics_classes_names}

interactors_classes_names_to_names = {
    k: v["name"] for k, v in settings["interactors_general_settings"].items()
}


nums_interactions_to_show = list(map(int, metric_evaluator.interactions_to_evaluate))


def generate_table_spec():
    res = "|"
    for i in range(1 + len(nums_interactions_to_show) * len(datasets_names)):
        res += "c"
        if i % (len(nums_interactions_to_show)) == 0:
            res += "|"
    return res


rtex_header = r"""
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
Dataset & %s \\""" % (
    generate_table_spec(),
    " & ".join(
        [
            r"\multicolumn{%d}{c|}{%s}" % (len(nums_interactions_to_show), i)
            for i in datasets_names
        ]
    ),
)
rtex_footer = r"""
\end{tabular}
\end{document}
"""
rtex = ""

datasets_metrics_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

datasets_metrics_users_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(list))
)

for dataset_name in datasets_names:

    for metric_class_name in metrics_classes_names:
        for agent_name in args.agents:
            agent_parameters = dataset_agents[dataset_name][agent_name]
            agent = utils.create_agent(agent_name, agent_parameters)
            agent_id = utils.get_agent_id(agent_name, agent_parameters)
            mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])
            dataset_parameters = settings["dataset_loaders"][dataset_name]
            metrics_evaluator_name = metric_evaluator.__class__.__name__
            parameters_agent_run = (
                utils.parameters_normalize(
                    constants.DATASET_PARAMETERS_PREFIX,
                    dataset_name,
                    dataset_parameters,
                )
                | utils.parameters_normalize(
                    constants.EVALUATION_POLICY_PARAMETERS_PREFIX,
                    evaluation_policy_name,
                    evaluation_policy_parameters,
                )
                | utils.parameters_normalize(
                    constants.AGENT_PARAMETERS_PREFIX, agent_name, agent_parameters
                )
            )

            parameters_evaluation_run = copy.copy(parameters_agent_run)
            parameters_evaluation_run |= utils.parameters_normalize(
                constants.METRIC_EVALUATOR_PARAMETERS_PREFIX, metric_evaluator_name, {}
            )
            parameters_evaluation_run |= utils.parameters_normalize(
                constants.METRIC_PARAMETERS_PREFIX, metric_class_name, {}
            )
            run = utils.already_ran(
                parameters_evaluation_run,
                mlflow.get_experiment_by_name(
                    settings["defaults"]["evaluation_experiment"]
                ).experiment_id,
            )
            # print(run)

            client = MlflowClient()
            artifact_path = client.download_artifacts(
                run.info.run_id, "evaluation.pickle"
            )
            # print(artifact_path)
            metric_values = pickle.load(open(artifact_path, "rb"))
            # print(metric_values)
            users_items_recommended = metric_values
            # agent = utils.create_agent_from_settings(agent_name,dataset_name,settings)
            # print(metric_values)
            # print(len(metric_values))

            datasets_metrics_values[dataset_name][metric_class_name][agent_name].extend(
                [
                    np.mean(list(metric_values[i].values()))
                    for i in range(len(nums_interactions_to_show))
                ]
            )
            datasets_metrics_users_values[dataset_name][metric_class_name][
                agent_name
            ].extend(
                np.array(
                    [
                        list(metric_values[i].values())
                        for i in range(len(nums_interactions_to_show))
                    ]
                )
            )

utility_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))
method_utility_scores = defaultdict(lambda: defaultdict(lambda: dict))
for num_interaction in range(len(nums_interactions_to_show)):
    for dataset_name in datasets_names:
        for metric_class_name in metrics_classes_names:
            for agent_name in args.agents:
                metric_max_value = np.max(
                    list(
                        map(
                            lambda x: x[num_interaction],
                            datasets_metrics_values[dataset_name][
                                metric_class_name
                            ].values(),
                        )
                    )
                )
                metric_min_value = np.min(
                    list(
                        map(
                            lambda x: x[num_interaction],
                            datasets_metrics_values[dataset_name][
                                metric_class_name
                            ].values(),
                        )
                    )
                )
                metric_value = datasets_metrics_values[dataset_name][metric_class_name][
                    agent_name
                ][num_interaction]
                utility_scores[dataset_name][metric_class_name][agent_name][
                    num_interaction
                ] = (metric_value - metric_min_value) / (
                    metric_max_value - metric_min_value
                )

for num_interaction in range(len(nums_interactions_to_show)):
    for dataset_name in datasets_names:
        for agent_name in args.agents:
            us = [
                utility_scores[dataset_name][metric_class_name][agent_name][
                    num_interaction
                ]
                * metrics_weights[metric_class_name]
                for metric_class_name in metrics_classes_names
            ]
            maut = np.sum(us)
            datasets_metrics_values[dataset_name]["MAUT"][agent_name].append(maut)
            datasets_metrics_users_values[dataset_name]["MAUT"][agent_name].append(
                np.array([maut] * 100)
            )

if args.dump:
    # with open('datasets_metrics_values.pickle','wb') as f:
    # pickle.dump(datasets_metrics_values,f)
    with open("datasets_metrics_values.pickle", "wb") as f:
        pickle.dump(json.loads(json.dumps(datasets_metrics_values)), f)
        # f.write(str(methods_users_hits))
# print(datasets_metrics_values['Yahoo Music']['MAUT'])

metrics_classes_names.append("MAUT")
metrics_names.append("MAUT")

datasets_metrics_gain = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(lambda: [""] * len(nums_interactions_to_show))
    )
)

datasets_metrics_best = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(lambda: [False] * len(nums_interactions_to_show))
    )
)
bullet_str = r"\textcolor[rgb]{0.7,0.7,0.0}{$\bullet$}"
triangle_up_str = r"\textcolor[rgb]{00,0.45,0.10}{$\blacktriangle$}"
triangle_down_str = r"\textcolor[rgb]{0.7,00,00}{$\blacktriangledown$}"

if args.type == "pairs":
    pool_of_methods_to_compare = [
        (args.agents[i], args.agents[i + 1]) for i in range(0, len(args.agents) - 1, 2)
    ]
else:
    pool_of_methods_to_compare = [[args.agents[i] for i in range(len(args.agents))]]
print(pool_of_methods_to_compare)
for dataset_name in datasets_names:
    for metric_class_name in metrics_classes_names:
        for i, num in enumerate(nums_interactions_to_show):
            for methods in pool_of_methods_to_compare:
                datasets_metrics_values_tmp = copy.deepcopy(datasets_metrics_values)
                methods_set = set(methods)
                for k1, v1 in datasets_metrics_values.items():
                    for k2, v2 in v1.items():
                        for k3, v3 in v2.items():
                            if k3 not in methods_set:
                                del datasets_metrics_values_tmp[k1][k2][k3]
                        # print(datasets_metrics_values_tmp[k1][k2])
                # datasets_metrics_values_tmp =datasets_metrics_values
                datasets_metrics_best[dataset_name][metric_class_name][
                    max(
                        datasets_metrics_values_tmp[dataset_name][
                            metric_class_name
                        ].items(),
                        key=lambda x: x[1][i],
                    )[0]
                ][i] = True
                if args.r == "lastmethod":
                    best_itr = methods[-1]
                elif args.r != None:
                    best_itr = args.r
                else:
                    best_itr = max(
                        datasets_metrics_values_tmp[dataset_name][
                            metric_class_name
                        ].items(),
                        key=lambda x: x[1][i],
                    )[0]
                best_itr_vals = datasets_metrics_values_tmp[dataset_name][
                    metric_class_name
                ].pop(best_itr)
                best_itr_val = best_itr_vals[i]
                second_best_itr = max(
                    datasets_metrics_values_tmp[dataset_name][
                        metric_class_name
                    ].items(),
                    key=lambda x: x[1][i],
                )[0]
                second_best_itr_vals = datasets_metrics_values_tmp[dataset_name][
                    metric_class_name
                ][second_best_itr]
                second_best_itr_val = second_best_itr_vals[i]
                # come back with value in dict
                datasets_metrics_values_tmp[dataset_name][metric_class_name][
                    best_itr
                ] = best_itr_vals

                best_itr_users_val = datasets_metrics_users_values[dataset_name][
                    metric_class_name
                ][best_itr][i]
                second_best_itr_users_val = datasets_metrics_users_values[dataset_name][
                    metric_class_name
                ][second_best_itr][i]

                try:
                    print(best_itr_users_val)
                    print(second_best_itr_users_val)
                    statistic, pvalue = scipy.stats.wilcoxon(
                        best_itr_users_val,
                        second_best_itr_users_val,
                    )
                except:
                    print("Wilcoxon error")
                    datasets_metrics_gain[dataset_name][metric_class_name][best_itr][
                        i
                    ] = bullet_str
                    continue

                if pvalue > 0.05:
                    datasets_metrics_gain[dataset_name][metric_class_name][best_itr][
                        i
                    ] = bullet_str
                else:
                    # print(best_itr,best_itr_val,second_best_itr,second_best_itr_val,methods)
                    if best_itr_val < second_best_itr_val:
                        datasets_metrics_gain[dataset_name][metric_class_name][
                            best_itr
                        ][i] = triangle_down_str
                    elif best_itr_val > second_best_itr_val:
                        datasets_metrics_gain[dataset_name][metric_class_name][
                            best_itr
                        ][i] = triangle_up_str
                    else:
                        datasets_metrics_gain[dataset_name][metric_class_name][
                            best_itr
                        ][i] = bullet_str

for metric_name, metric_class_name in zip(metrics_names, metrics_classes_names):
    rtex += utils.generate_metric_interactions_header(
        nums_interactions_to_show, len(datasets_names), metric_name
    )
    for agent_name in args.agents:
        rtex += "%s & " % (utils.get_agent_pretty_name(agent_name, settings))
        rtex += " & ".join(
            [
                " & ".join(
                    map(
                        lambda x, y, z: (r"\textbf{" if z else "")
                        + f"{x:.3f}{y}"
                        + (r"}" if z else ""),
                        datasets_metrics_values[dataset_name][metric_class_name][
                            agent_name
                        ],
                        datasets_metrics_gain[dataset_name][metric_class_name][
                            agent_name
                        ],
                        datasets_metrics_best[dataset_name][metric_class_name][
                            agent_name
                        ],
                    )
                )
                for dataset_name in datasets_names
            ]
        )
        rtex += r"\\\hline" + "\n"

res = rtex_header + rtex + rtex_footer

tmp = "_".join([dataset_name for dataset_name in datasets_names])
tex_path = os.path.join(
    settings["defaults"]["data_dir"],
    settings["defaults"]["tex_dir"],
    f"table_{tmp}.tex",
)
utils.create_path_to_file(tex_path)
open(
    tex_path,
    "w+",
).write(res)
pdf_path = os.path.join(
    settings["defaults"]["data_dir"], settings["defaults"]["pdf_dir"]
)
utils.create_path_to_file(pdf_path)
# print(f"latexmk -pdf -interaction=nonstopmode -output-directory={pdf_path} {tex_path}")
os.system(
    f'latexmk -pdflatex=pdflatex -pdf -interaction=nonstopmode -output-directory="{pdf_path}" "{tex_path}"'
)
