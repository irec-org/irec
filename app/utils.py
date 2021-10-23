from os.path import dirname, realpath, sep, pardir
import os
import sys

sys.path.append(dirname(dirname(realpath(__file__))))
import re
from app import errors
import pickle
import mlflow.tracking
import mlflow.entities
import mlflow
from mlflow.tracking import MlflowClient
import json
from collections import defaultdict
from pathlib import Path

from irec.utils.dataset import TrainTestDataset
import collections
from app import constants
import matplotlib.ticker as mtick
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, realpath, sep, pardir
import irec.action_selection_policies
import irec.agents
import irec.value_functions
import irec.evaluation_policies
from irec.metric_evaluators import (
    CumulativeInteractionMetricEvaluator,
    InteractionMetricEvaluator,
    CumulativeMetricEvaluator,
    UserCumulativeInteractionMetricEvaluator,
)
import copy
import os.path
import collections.abc
import pandas as pd

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


class Singleton:
    def __init__(self, decorated):
        self._decorated = decorated

    def instance(self):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError("Singletons must be accessed through `instance()`.")


@Singleton
class Settings:
    def __init__(self) -> None:
        self.settings = None
        pass

    def load_settings(self):
        self.settings = load_settings()


def gen_dict_extract(key, var):

    if hasattr(var, "items"):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, dict):
                for result in gen_dict_extract(key, v):
                    yield result
            elif isinstance(v, list):
                for d in v:
                    for result in gen_dict_extract(key, d):
                        yield result


def update_nested_dict(d, u):
    for k, dv in d.items():
        if k in u:
            uv = u[k]
            if isinstance(dv, collections.abc.Mapping):
                d[k] = update_nested_dict(uv.get(k, {}), dv)
            else:
                d[k] = uv
    return d


def class2dict(instance):
    if not hasattr(instance, "__dict__"):
        return instance
    new_subdic = dict(vars(instance))
    for key, value in new_subdic.items():
        new_subdic[key] = class2dict(value)
    return new_subdic


def generate_table_spec(nums_interactions_to_show, num_datasets_preprocessors):
    res = "|"
    for i in range(1 + len(nums_interactions_to_show) * num_datasets_preprocessors):
        res += "c"
        if i % (len(nums_interactions_to_show)) == 0:
            res += "|"
    return res


def generate_datasets_line(nums_interactions_to_show, preprocessors_names):
    return " & ".join(
        [
            r"\multicolumn{%d}{c|}{%s}" % (len(nums_interactions_to_show), i)
            for i in preprocessors_names
        ]
    )


def generate_metrics_header_line(
    nums_interactions_to_show, num_preprocessors, metric_name
):
    return " & ".join(
        map(
            lambda x: r"\multicolumn{%d}{c|}{%s}" % (len(nums_interactions_to_show), x),
            [metric_name] * num_preprocessors,
        )
    )


def generate_interactions_header_line(nums_interactions_to_show, num_preprocessors):
    return " & ".join(
        [" & ".join(map(str, nums_interactions_to_show))] * num_preprocessors
    )


def generate_metric_interactions_header(
    nums_interactions_to_show, num_preprocessors, metric_name
):
    btex = LATEX_TABLE_METRICS_INTERACTIONS_HEADER % (
        generate_metrics_header_line(
            nums_interactions_to_show, num_preprocessors, metric_name
        ),
        generate_interactions_header_line(nums_interactions_to_show, num_preprocessors),
    )
    return btex


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if v == {}:
            items.append((new_key, v))
        elif isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def rec_defaultdict(d=None):
    if d == None:
        return defaultdict(rec_defaultdict)
    else:
        return defaultdict(rec_defaultdict, d)


def defaultify(d):
    if not isinstance(d, dict):
        return d
    return defaultdict(lambda: dict(), {k: defaultify(v) for k, v in d.items()})


def _do_nothing(d):
    return d


def load_settings():
    d = dict()
    loader = yaml.SafeLoader
    # d["agents_preprocessor_parameters"] = yaml.load(
    # open(
    # dirname(realpath(__file__))
    # + sep
    # + "settings"
    # + sep
    # + "agents_preprocessor_parameters.yaml"
    # ),
    # Loader=loader,
    # )
    # d["agents_preprocessor_parameters"] = _do_nothing(
    # d["agents_preprocessor_parameters"]
    # )

    d["interactors_general_settings"] = yaml.load(
        open(
            dirname(realpath(__file__))
            + sep
            + "settings"
            + sep
            + "interactors_general_settings.yaml"
        ),
        Loader=loader,
    )

    # d["agents_search_parameters"] = yaml.load(
    # open(
    # dirname(realpath(__file__))
    # + sep
    # + "settings"
    # + sep
    # + "agents_search_parameters.yaml"
    # ),
    # Loader=loader,
    # )

    d["evaluation_policies"] = yaml.load(
        open(
            dirname(realpath(__file__))
            + sep
            + "settings"
            + sep
            + "evaluation_policies.yaml"
        ),
        Loader=loader,
    )

    d["dataset_loaders"] = yaml.load(
        open(
            dirname(realpath(__file__))
            + sep
            + "settings"
            + sep
            + "dataset_loaders.yaml"
        ),
        Loader=loader,
    )

    d["agents"] = yaml.load(
        open(dirname(realpath(__file__)) + sep + "settings" + sep + "agents.yaml"),
        Loader=loader,
    )

    d["defaults"] = yaml.load(
        open(dirname(realpath(__file__)) + sep + "settings" + sep + "defaults.yaml"),
        Loader=loader,
    )

    d["metric_evaluators"] = yaml.load(
        open(
            dirname(realpath(__file__))
            + sep
            + "settings"
            + sep
            + "metric_evaluators.yaml"
        ),
        Loader=loader,
    )

    # with open(
    # dirname(realpath(__file__)) + sep + "settings" + sep +
    # "datasets_preprocessors_parameters.yaml") as f:
    # d['datasets_preprocessors_parameters'] = yaml.load(f, Loader=loader)
    # d['datasets_preprocessors_parameters'] = {
    # k: {
    # **setting,
    # **{
    # 'name': k
    # }
    # } for k, setting in d['datasets_preprocessors_parameters'].items()
    # }
    with open(
        dirname(realpath(__file__)) + sep + "settings" + sep + "defaults.yaml"
    ) as f:
        d["defaults"] = yaml.load(f, Loader=loader)
    return d


def load_settings_to_parser(settings, parser):
    settings_flatten = flatten_dict(settings)
    for k, v in settings_flatten.items():
        # agent_group = parser.add_argument_group(agent_name)
        # parser.add_argument(f"--{k}", default=v, type=yaml.safe_load)
        parser.add_argument(f"--{k}", default=v, type=type(v))
        # parser.add_argument(f'--{k}')


def sync_settings_from_args(settings, args, sep="."):
    settings = copy.deepcopy(settings)
    args_dict = vars(args)
    settings_flatten = flatten_dict(settings)
    for i in set(args_dict.keys()).intersection(set(settings_flatten.keys())):
        tmp = settings
        for j in i.split(sep)[:-1]:
            tmp = tmp[j]
        tmp[i.split(sep)[-1]] = args_dict[i]
    return settings


def plot_similar_items(ys, method1, method2, title=None):
    fig, ax = plt.subplots()
    ax.plot(np.sort(ys)[::-1], linewidth=2, color="k")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.set_title()
    ax.set_title(title)
    ax.set_xlabel("Users Rank")
    ax.set_ylabel("Similar items (%)")
    return fig


def get_experiment_run_id(dm, evaluation_policy, itr_id):
    return os.path.join(dm.get_id(), evaluation_policy.get_id(), itr_id)


def run_interactor(
    agent,
    traintest_dataset: TrainTestDataset,
    evaluation_policy: irec.evaluation_policies.EvaluationPolicy,
    settings,
    forced_run,
):

    mlflow.set_experiment(settings["defaults"]["agent_experiment"])

    run = get_agent_run(settings)
    if forced_run == False and run != None:
        print(
            "Already executed {} in {}".format(
                get_agent_run_parameters(settings),
                settings["defaults"]["agent_experiment"],
            )
        )
        return
    with mlflow.start_run() as run:
        log_custom_parameters(get_agent_run_parameters(settings))

        interactions, acts_info = evaluation_policy.evaluate(
            agent, traintest_dataset.train, traintest_dataset.test
        )

        fname = "./tmp/interactions.pickle"
        log_custom_artifact(fname, interactions)
        fname = "./tmp/acts_info.pickle"
        log_custom_artifact(fname, acts_info)
        # create_path_to_file(fname)
        # with open(fname,mode='wb') as f:
        # pickle.dump(history_items_recommended,f)
        # mlflow.log_artifact(f.name)


def get_agent_id(agent_name, agent_parameters):
    # agent_dict = class2dict(agent)
    return agent_name + "_" + json.dumps(agent_parameters, separators=(",", ":"))


# def get_agent_id(agent, template_parameters):
# agent_dict = class2dict(agent)
# new_agent_settings = update_nested_dict(template_parameters, agent_dict)
# return agent.name + '_' + json.dumps(new_agent_settings,
# separators=(',', ':'))


def get_agent_id_from_settings(agent, settings):
    agent_settings = next(
        gen_dict_extract(agent.name, settings["agents_preprocessor_parameters"])
    )
    # agent_settings = copy.copy(settings['agents_preprocessor_parameters'][dataset_preprocessor_name][agent.name])
    return get_agent_id(agent, agent_settings)


def create_action_selection_policy(action_selection_policy_settings):
    action_selection_policy_name = list(action_selection_policy_settings.keys())[0]
    action_selection_policy_parameters = list(
        action_selection_policy_settings.values()
    )[0]
    action_selection_policy = eval(
        "irec.action_selection_policies." + action_selection_policy_name
    )(**action_selection_policy_parameters)

    if isinstance(action_selection_policy, irec.action_selection_policies.ASPReranker):
        action_selection_policy.rule = create_value_function(
            action_selection_policy.rule
        )
    return action_selection_policy


def create_value_function(value_function_settings):
    value_function_name = list(value_function_settings.keys())[0]
    value_function_parameters = list(value_function_settings.values())[0]

    if value_function_name in [
        "OurMethodRandom",
        "OurMethodRandPopularity",
        "OurMethodEntropy",
        "OurMethodPopularity",
        "OurMethodOne",
        "OurMethodZero",
    ]:
        exec("import irec.value_functions.OurMethodInit")
        value_function = eval(
            "irec.value_functions.OurMethodInit.{}".format(value_function_name)
        )(**value_function_parameters)
    else:
        exec("import irec.value_functions.{}".format(value_function_name))
        value_function = eval(
            "irec.value_functions.{}.{}".format(
                value_function_name, value_function_name
            )
        )(**value_function_parameters)
    return value_function


def create_agent(agent_name, agent_settings):
    agent_class_parameters = {}
    agent_class_name = list(agent_settings.keys())[0]
    agent_parameters = list(agent_settings.values())[0]
    agent_class = eval("irec.agents." + agent_class_name)
    action_selection_policy = create_action_selection_policy(
        agent_parameters["action_selection_policy"]
    )
    # action_selection_policy = eval('irec.action_selection_policies.'+action_selection_policy_name)(**action_selection_policy_parameters)

    # value_function_name = list(agent_parameters['value_function'].keys())[0]
    # value_function_parameters = list(agent_parameters['value_function'].values())[0]
    value_function = create_value_function(agent_parameters["value_function"])
    agents = []
    if agent_name in [
        "NaiveEnsemble",
        "TSEnsemble_Pop",
        "TSEnsemble_PopEnt",
        "TSEnsemble_Entropy",
        "TSEnsemble_Random",
    ]:
        for _agent in agent_parameters["agents"]:
            # print(_agent)
            new_agent = create_agent(list(_agent.keys())[0], list(_agent.values())[0])
            agents.append(new_agent)
        agent_class_parameters["agents"] = agents
    agent_class_parameters.update(
        {
            "action_selection_policy": action_selection_policy,
            "value_function": value_function,
            "name": agent_name,
        }
    )
    # print(agent_class_parameters)
    return agent_class(**agent_class_parameters)


def create_agent_from_settings(agent_name, dataset_preprocessor_name, settings):
    agent_settings = settings["agents_preprocessor_parameters"][
        dataset_preprocessor_name
    ][agent_name]
    agent = create_agent(agent_name, agent_settings)
    return agent


def default_to_regular(d):
    if isinstance(d, (defaultdict, dict)):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def get_agent_pretty_name(agent_name, settings):
    return settings["interactors_general_settings"][agent_name]["name"]


def nested_dict_to_df(values_dict):
    def flatten_dict(nested_dict):
        res = {}
        if isinstance(nested_dict, dict):
            for k in nested_dict:
                flattened_dict = flatten_dict(nested_dict[k])
                for key, val in flattened_dict.items():
                    key = list(key)
                    key.insert(0, k)
                    res[tuple(key)] = val
        else:
            res[()] = nested_dict
        return res

    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df


def create_path_to_file(file_name):
    Path("/".join(file_name.split("/")[:-1])).mkdir(parents=True, exist_ok=True)


def _get_params(run):
    """Converts [mlflow.entities.Param] to a dictionary of {k: v}."""
    return run.data.params


def already_ran(parameters, experiment_id):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    # print('Exp',experiment_id)
    all_run_infos = mlflow.list_run_infos(
        experiment_id, order_by=["attribute.end_time DESC"]
    )
    for run_info in all_run_infos:
        # print(run_info)

        full_run = mlflow.get_run(run_info.run_uuid)
        run_params = _get_params(full_run)
        match_failed = False
        # print(parameters)
        # print(run_params)
        for param_key, param_value in parameters.items():
            run_value = run_params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.status != "FINISHED":
            print(
                (
                    "Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)"
                )
                % (run_info.run_uuid, run_info.status)
            )
            continue
        return mlflow.get_run(run_info.run_uuid)
    # raise IndexError("Could not find the run with the given parameters.")
    return None


def dict_parameters_normalize(category, settings: dict):
    settings = flatten_dict({category: settings})
    return settings


def parameters_normalize(category, name, settings: dict):
    t1 = {**dict_parameters_normalize(category, settings), **{category: name}}
    return {str(k): str(v) for k, v in t1.items()}


def log_custom_parameters(settings: dict) -> None:
    # settings = {name: settings}
    # settings = flatten_dict(settings)
    # settings = {re.sub(' ','_',k):v for k,v in settings.items()}
    # mlflow.log_param(field, name)
    for k, v in settings.items():
        mlflow.log_param(k, v)


# def log_dataset_parameters(dataset_name, dataset_loader_settings: dict):
# log_custom_parameters('dataset', dataset_name, {'dataset':dataset_loader_settings})


# def log_category_parameters(x, name, settings: dict):
# log_custom_parameters(name)


# def log_evaluation_policy_parameters(name, settings: dict):
# cn = 'evaluation_policy'
# log_custom_parameters(cn, name, {cn:settings})


import secrets


def log_custom_artifact(fname, obj):
    fnametmp = f"./tmp/{secrets.token_urlsafe(16)}/{fname}"
    create_path_to_file(fnametmp)
    with open(fnametmp, mode="wb") as f:
        pickle.dump(obj, f)
        # f.flush()
    mlflow.log_artifact(fnametmp)


def load_dataset_experiment(settings):
    run = already_ran(
        parameters_normalize(
            constants.DATASET_PARAMETERS_PREFIX,
            settings["defaults"]["dataset_loader"],
            settings["dataset_loaders"][settings["defaults"]["dataset_loader"]],
        ),
        mlflow.get_experiment_by_name(
            settings["defaults"]["dataset_experiment"]
        ).experiment_id,
    )

    client = MlflowClient()
    artifact_path = client.download_artifacts(run.info.run_id, "dataset.pickle")
    traintest_dataset = pickle.load(open(artifact_path, "rb"))
    return traintest_dataset


def run_agent(traintest_dataset, settings, forced_run):

    dataset_loader_parameters = settings["dataset_loaders"][
        settings["defaults"]["dataset_loader"]
    ]

    evaluation_policy_parameters = settings["evaluation_policies"][
        settings["defaults"]["evaluation_policy"]
    ]
    evaluation_policy = eval(
        "irec.evaluation_policies." + settings["defaults"]["evaluation_policy"]
    )(**evaluation_policy_parameters)

    mlflow.set_experiment(settings["defaults"]["dataset_experiment"])

    agent_parameters = settings["agents"][settings["defaults"]["agent"]]
    agent = create_agent(settings["defaults"]["agent"], agent_parameters)
    # agent_id = utils.get_agent_id(agent_name, parameters)
    run_interactor(
        agent=agent,
        traintest_dataset=traintest_dataset,
        evaluation_policy=evaluation_policy,
        settings=settings,
        forced_run=forced_run,
    )


def evaluate_itr(dataset, settings, forced_run):
    run = get_evaluation_run(settings)
    if forced_run == False and run != None:
        print(
            "Already executed {} in {}".format(
                get_evaluation_run_parameters(settings),
                settings["defaults"]["agent_experiment"],
            )
        )
        return
    agent_parameters = settings["agents"][settings["defaults"]["agent"]]

    dataset_parameters = settings["dataset_loaders"][
        settings["defaults"]["dataset_loader"]
    ]

    evaluation_policy_parameters = settings["evaluation_policies"][
        settings["defaults"]["evaluation_policy"]
    ]
    # evaluation_policy = eval(
    # "irec.evaluation_policies." + settings["defaults"]["evaluation_policy"]
    # )(**evaluation_policy_parameters)

    metric_evaluator_parameters = settings["metric_evaluators"][
        settings["defaults"]["metric_evaluator"]
    ]

    metric_class = eval("irec.metrics." + settings["defaults"]["metric"])
    print(settings["defaults"]["metric_evaluator"], metric_evaluator_parameters)
    metric_evaluator = eval(
        "irec.metric_evaluators." + settings["defaults"]["metric_evaluator"]
    )(dataset, **metric_evaluator_parameters)

    mlflow.set_experiment(settings["defaults"]["agent_experiment"])
    # print(parameters_agent_run)
    run = get_agent_run(settings)

    if run == None:
        print("Could not find agent run")
        return
    client = MlflowClient()
    artifact_path = client.download_artifacts(run.info.run_id, "interactions.pickle")
    with open(artifact_path, "rb") as f:
        interactions = pickle.load(f)

    mlflow.set_experiment(settings["defaults"]["agent_experiment"])
    # print(parameters_agent_run)
    run = get_agent_run(settings)
    client = MlflowClient()
    artifact_path = client.download_artifacts(run.info.run_id, "interactions.pickle")
    # print(artifact_path)
    interactions = pickle.load(open(artifact_path, "rb"))
    metric_values = metric_evaluator.evaluate(
        metric_class,
        interactions,
    )
    with mlflow.start_run(run_id=run.info.run_id) as run:
        if isinstance(metric_evaluator, UserCumulativeInteractionMetricEvaluator):
            mlflow.log_metric(
                metric_class.__name__, np.mean(list(metric_values[-1].values()))
            )
        elif isinstance(metric_evaluator, InteractionMetricEvaluator):
            pass
        elif isinstance(metric_evaluator, CumulativeMetricEvaluator):
            pass

    mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])
    parameters_evaluation_run = get_evaluation_run_parameters(settings)

    with mlflow.start_run() as run:
        log_custom_parameters(parameters_evaluation_run)
        if isinstance(metric_evaluator, UserCumulativeInteractionMetricEvaluator):
            mlflow.log_metric(
                metric_class.__name__, np.mean(list(metric_values[-1].values()))
            )
        elif isinstance(metric_evaluator, InteractionMetricEvaluator):
            pass
        elif isinstance(metric_evaluator, CumulativeMetricEvaluator):
            pass
        # print(metric_values)
        log_custom_artifact("evaluation.pickle", metric_values)


def load_evaluation_experiment(settings):
    mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])
    dataset_parameters = settings["dataset_loaders"][
        settings["defaults"]["dataset_loader"]
    ]
    # metrics_evaluator_name = settings["defaults"]["metric_evaluator"]
    run = get_evaluation_run(settings)

    if run == None:
        raise errors.EvaluationRunNotFoundError("Could not find evaluation run")
    client = MlflowClient()
    artifact_path = client.download_artifacts(run.info.run_id, "evaluation.pickle")
    # print(artifact_path)
    metric_values = pickle.load(open(artifact_path, "rb"))
    return metric_values


def get_agent_run_parameters(settings):
    parameters_agent_run = {
        **parameters_normalize(
            constants.DATASET_PARAMETERS_PREFIX,
            settings["defaults"]["dataset_loader"],
            settings["dataset_loaders"][settings["defaults"]["dataset_loader"]],
        ),
        **parameters_normalize(
            constants.EVALUATION_POLICY_PARAMETERS_PREFIX,
            settings["defaults"]["evaluation_policy"],
            settings["evaluation_policies"][settings["defaults"]["evaluation_policy"]],
        ),
        **parameters_normalize(
            constants.AGENT_PARAMETERS_PREFIX,
            settings["defaults"]["agent"],
            settings["agents"][settings["defaults"]["agent"]],
        ),
    }
    return parameters_agent_run


def get_evaluation_run_parameters(settings):
    parameters_agent_run = get_agent_run_parameters(settings)
    # parameters_evaluation_run = copy.copy(parameters_agent_run)
    parameters_evaluation_run = {
        **parameters_agent_run,
        **parameters_normalize(
            constants.METRIC_EVALUATOR_PARAMETERS_PREFIX,
            settings["defaults"]["metric_evaluator"],
            settings["metric_evaluators"][settings["defaults"]["metric_evaluator"]],
        ),
        **parameters_normalize(
            constants.METRIC_PARAMETERS_PREFIX,
            settings["defaults"]["metric"],
            {},
        ),
    }
    return parameters_evaluation_run


# def get_agent_run(settings):
# parameters_evaluation_run = get_parameters_agent_run(settings)

# # parameters_evaluation_run |= parameters_normalize(
# # constants.METRIC_PARAMETERS_PREFIX, settings["defaults"]["metric"], {}
# # )
# run = already_ran(
# parameters_evaluation_run,
# mlflow.get_experiment_by_name(
# settings["defaults"]["evaluation_experiment"]
# ).experiment_id,
# )
# return run
def get_agent_run(settings):
    agent_run_parameters = get_agent_run_parameters(settings)
    run = already_ran(
        agent_run_parameters,
        mlflow.get_experiment_by_name(
            settings["defaults"]["agent_experiment"]
        ).experiment_id,
    )
    return run


def get_evaluation_run(settings):
    evaluation_run_parameters = get_evaluation_run_parameters(settings)
    run = already_ran(
        evaluation_run_parameters,
        mlflow.get_experiment_by_name(
            settings["defaults"]["evaluation_experiment"]
        ).experiment_id,
    )
    return run


def unflatten_dict(d, sep="."):
    result_dict = dict()
    for key, value in d.items():
        parts = key.split(sep)
        d = result_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return result_dict
