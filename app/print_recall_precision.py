#!/usr/bin/python3


def flatten_dict(deep_dict):
    def do_flatten(deep_dict, current_key):
        for key, value in deep_dict.items():
            # the key will be a flattened tuple
            # but the type of `key` is not touched
            new_key = current_key + (key,)
            # if we have a dict, we recurse
            if isinstance(value, dict):
                yield from do_flatten(value, new_key)
            else:
                yield (new_key, value)

    return dict(do_flatten(deep_dict, ()))


from os.path import dirname, realpath, sep, pardir
import mlflow
from mlflow.tracking import MlflowClient
import pickle
import os
import sys
import json

from pandas.core.indexes.extension import make_wrapped_arith_op
import utils
import pandas as pd

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
from irec.environment.dataset import Dataset
from irec import metrics
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
settings = utils.load_settings()
utils.load_settings_to_parser(settings, parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings, args)

plt.rcParams["axes.prop_cycle"] = cycler(color="krbgmyc")
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 18

dataset_agents = yaml.load(
    open("./settings/dataset_agents.yaml"), Loader=yaml.SafeLoader
)

metrics_classes = [eval("metrics." + i) for i in args.metrics]

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

interactors_classes_names_to_names = {
    k: v["name"] for k, v in settings["agents_general_settings"].items()
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

datasets_metrics_values = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])

runs_infos = mlflow.list_run_infos(
    mlflow.get_experiment_by_name(
        settings["defaults"]["evaluation_experiment"]
    ).experiment_id,
    order_by=["attribute.end_time DESC"],
)

for dataset_loader_name in datasets_names:
    for metric_class_name in metrics_classes_names:
        for agent_name in args.agents:
            settings["defaults"]["dataset_loader"] = dataset_loader_name
            settings["defaults"]["agent"] = agent_name
            agent_parameters = dataset_agents[dataset_loader_name][agent_name]
            settings["agents"][agent_name] = agent_parameters
            settings["defaults"]["metric"] = metric_class_name
            agent = utils.create_agent(agent_name, agent_parameters)
            agent_id = utils.get_agent_id(agent_name, agent_parameters)
            dataset_parameters = settings["dataset_loaders"][dataset_loader_name]
            metrics_evaluator_name = metric_evaluator.__class__.__name__
            parameters_agent_run = utils.get_agent_run_parameters(settings)
            parameters_evaluation_run = utils.get_evaluation_run_parameters(settings)

            mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])
            print(dataset_loader_name, agent_name)
            run = utils.already_ran(
                parameters_evaluation_run,
                mlflow.get_experiment_by_name(
                    settings["defaults"]["evaluation_experiment"]
                ).experiment_id,
                runs_infos,
            )

            client = MlflowClient()
            artifact_path = client.download_artifacts(
                run.info.run_id, "evaluation.pickle"
            )
            metric_values = pickle.load(open(artifact_path, "rb"))
            users_items_recommended = metric_values

            datasets_metrics_values[dataset_loader_name][metric_class_name][
                interactors_classes_names_to_names[agent_name]
            ] = {
                str(nums_interactions_to_show[i]): [np.mean(list(metric_values[i].values()))]
                for i in range(len(nums_interactions_to_show))
            }

df = pd.DataFrame.from_dict(flatten_dict(datasets_metrics_values))
df=df.stack().reset_index(level=0, drop=True)
# df= df.stack().unstack(level=1)
# print(df)
# print(df)
markers = ["o","^","*","d","s"]
colors = ['k','tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
for metric_class_name in metrics_classes_names:
    methods=[]
    for dataset_loader_name in datasets_names:
        # ds_metric_df = df[dataset_loader_name][metric_class_name]
        ds_recall_df = df[dataset_loader_name]['Recall']
        methods.extend(ds_recall_df.loc[str(max(nums_interactions_to_show))].T.sort_values(ascending=False).index[:5])
    methods=list(set(methods))
    # sort(methods)
    methods.sort()
    methods=list(reversed(methods))
    mcolors = {methods[i]: colors[i] for i in range(len(methods))}
    lines = {}

    for dataset_loader_name in datasets_names:
        ds_metric_df = df[dataset_loader_name][metric_class_name]
        ds_recall_df = df[dataset_loader_name]['Recall']
        ds_metric_df= ds_metric_df[ds_recall_df.loc[str(max(nums_interactions_to_show))].T.sort_values(ascending=False).index[:5]]

        # print(ds_metric_df)
        fig, ax = plt.subplots()

        # ds_metric_df.plot()
        for i,marker in zip(ds_metric_df.columns,markers):
            (line,)= ax.plot(ds_metric_df[i],marker=marker,linestyle='dashed',markersize=7,color=mcolors[i])
            lines[i]=line
        tmp = dataset_loader_name + "_" + metric_class_name
        # ax.legend(ds_metric_df.columns,frameon=False)
        # ax.legend(ds_metric_df.columns,bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        # mode="expand", borderaxespad=0, ncol=3)
        # ax.set_ylim(ds_metric_df.min().min(),ds_metric_df.max().max()*1.35)
        ax.set_xticks(list(range(len(nums_interactions_to_show))))
        ax.set_xticklabels(nums_interactions_to_show)
        ax.set_xlabel('Top-k')
        ax.set_ylabel(metric_class_name)

        path = os.path.join(
            settings["defaults"]["data_dir"],
            settings["defaults"]["general_dir"],
            f"{tmp}",
        )
        fig.savefig(path+'.png', bbox_inches="tight")
        fig.savefig(path+'.eps', bbox_inches="tight")


    figlegend = plt.figure()
    figlegend.legend(
        list(lines.values()),
        methods,
        loc="center",
        ncol=len(methods)
    )

    path = os.path.join(
    settings["defaults"]["data_dir"],
    settings["defaults"]["general_dir"],
    "legend",
    )
    figlegend.savefig(
        path+'.png',
        bbox_inches="tight",
    )
    figlegend.savefig(
        path+'.eps',
        bbox_inches="tight",
    )
