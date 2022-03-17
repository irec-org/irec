from os.path import sep
import os
from app import errors
import pickle
import yaml
import secrets
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
from os.path import sep
import irec.value_functions
from irec.evaluation_policies.EvaluationPolicy import EvaluationPolicy
import irec.evaluation_policies
from irec.utils.Factory import (
    AgentFactory,
)
import scipy
from irec.metric_evaluators.InteractionMetricEvaluator import InteractionMetricEvaluator
from irec.metric_evaluators.CumulativeMetricEvaluator import CumulativeMetricEvaluator
from irec.metric_evaluators.UserCumulativeInteractionMetricEvaluator import (
    UserCumulativeInteractionMetricEvaluator,
)
import copy
import os.path
import collections.abc
import pandas as pd
import scipy.stats


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
    if d is None:
        return defaultdict(rec_defaultdict)
    else:
        return defaultdict(rec_defaultdict, d)


def defaultify(d):
    if not isinstance(d, dict):
        return d
    return defaultdict(lambda: dict(), {k: defaultify(v) for k, v in d.items()})


def _do_nothing(d):
    return d


def load_settings(workdir):
    d = dict()
    loader = yaml.SafeLoader

    d["agents_general_settings"] = yaml.load(
        open(workdir + sep + "settings" + sep + "agents_general_settings.yaml"),
        Loader=loader,
    )

    d["evaluation_policies"] = yaml.load(
        open(workdir + sep + "settings" + sep + "evaluation_policies.yaml"),
        Loader=loader,
    )

    d["dataset_loaders"] = yaml.load(
        open(workdir + sep + "settings" + sep + "dataset_loaders.yaml"),
        Loader=loader,
    )

    d["agents"] = yaml.load(
        open(workdir + sep + "settings" + sep + "agents.yaml"),
        Loader=loader,
    )

    d["defaults"] = yaml.load(
        open(workdir + sep + "settings" + sep + "defaults.yaml"),
        Loader=loader,
    )

    d["metric_evaluators"] = yaml.load(
        open(workdir + sep + "settings" + sep + "metric_evaluators.yaml"),
        Loader=loader,
    )

    # with open(
    # workdir + sep + "settings" + sep +
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
    with open(workdir + sep + "settings" + sep + "defaults.yaml") as f:
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
    evaluation_policy: EvaluationPolicy,
    settings,
    forced_run,
):

    mlflow.set_experiment(settings["defaults"]["agent_experiment"])

    run = get_agent_run(settings)
    if forced_run is False and run is not None:
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


def create_agent_from_settings(agent_name, dataset_preprocessor_name, settings):
    agent_settings = settings["agents_preprocessor_parameters"][
        dataset_preprocessor_name
    ][agent_name]

    agent = AgentFactory().create(agent_name, agent_settings)
    return agent


def default_to_regular(d):
    if isinstance(d, (defaultdict, dict)):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def get_agent_pretty_name(agent_name, settings):
    return settings["agents_general_settings"][agent_name]["name"]


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


def already_ran(parameters, experiment_id, runs_infos):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    # print('Exp',experiment_id)
    all_run_infos = runs_infos
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
        runs_infos=mlflow.list_run_infos(
            mlflow.get_experiment_by_name(
                settings["defaults"]["dataset_experiment"]
            ).experiment_id,
            order_by=["attribute.end_time DESC"],
        ),
    )

    client = MlflowClient()
    artifact_path = client.download_artifacts(run.info.run_id, "dataset.pickle")
    traintest_dataset = pickle.load(open(artifact_path, "rb"))
    return traintest_dataset


def run_agent(traintest_dataset, settings, forced_run):

    # dataset_loader_parameters = settings["dataset_loaders"][
    # settings["defaults"]["dataset_loader"]
    # ]

    evaluation_policy_name = settings["defaults"]["evaluation_policy"]
    evaluation_policy_parameters = settings["evaluation_policies"][
        evaluation_policy_name
    ]

    # exec("import irec.value_functions.{}".format(value_function_name))
    #         value_function = eval(
    #             "irec.value_functions.{}.{}".format(
    #                 value_function_name, value_function_name
    #             )
    #         )(**value_function_parameters)
    exec(
        f"from irec.evaluation_policies.{evaluation_policy_name} import {evaluation_policy_name}"
    )
    evaluation_policy = eval(evaluation_policy_name)(**evaluation_policy_parameters)

    mlflow.set_experiment(settings["defaults"]["dataset_experiment"])

    agent_parameters = settings["agents"][settings["defaults"]["agent"]]
    agent = AgentFactory().create(settings["defaults"]["agent"], agent_parameters)
    run_interactor(
        agent=agent,
        traintest_dataset=traintest_dataset,
        evaluation_policy=evaluation_policy,
        settings=settings,
        forced_run=forced_run,
    )


def evaluate_itr(dataset, settings, forced_run):
    run = get_evaluation_run(settings)
    if forced_run is False and run is not None:
        print(
            "Already executed {} in {}".format(
                get_evaluation_run_parameters(settings),
                settings["defaults"]["agent_experiment"],
            )
        )
        return
    # agent_parameters = settings["agents"][settings["defaults"]["agent"]]

    # dataset_parameters = settings["dataset_loaders"][
    # settings["defaults"]["dataset_loader"]
    # ]

    # evaluation_policy_parameters = settings["evaluation_policies"][
    # settings["defaults"]["evaluation_policy"]
    # ]
    # evaluation_policy = eval(
    # "irec.evaluation_policies." + settings["defaults"]["evaluation_policy"]
    # )(**evaluation_policy_parameters)

    metric_evaluator_parameters = settings["metric_evaluators"][
        settings["defaults"]["metric_evaluator"]
    ]

    metric_class = eval("irec.metrics." + settings["defaults"]["metric"])
    print(settings["defaults"]["metric_evaluator"], metric_evaluator_parameters)

    metric_evaluator_name = settings["defaults"]["metric_evaluator"]
    exec(
        f"from irec.metric_evaluators.{metric_evaluator_name} import {metric_evaluator_name}"
    )
    metric_evaluator = eval(metric_evaluator_name)(
        dataset, **metric_evaluator_parameters
    )

    mlflow.set_experiment(settings["defaults"]["agent_experiment"])
    # print(parameters_agent_run)
    run = get_agent_run(settings)

    if run is None:
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
        print(metric_evaluator, UserCumulativeInteractionMetricEvaluator)
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
    # dataset_parameters = settings["dataset_loaders"][
    # settings["defaults"]["dataset_loader"]
    # ]
    # metrics_evaluator_name = settings["defaults"]["metric_evaluator"]
    run = get_evaluation_run(settings)

    if run is None:
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
        runs_infos=mlflow.list_run_infos(
            mlflow.get_experiment_by_name(
                settings["defaults"]["agent_experiment"]
            ).experiment_id,
            order_by=["attribute.end_time DESC"],
        ),
    )
    return run


def get_evaluation_run(settings):
    evaluation_run_parameters = get_evaluation_run_parameters(settings)
    run = already_ran(
        evaluation_run_parameters,
        mlflow.get_experiment_by_name(
            settings["defaults"]["evaluation_experiment"]
        ).experiment_id,
        runs_infos=mlflow.list_run_infos(
            mlflow.get_experiment_by_name(
                settings["defaults"]["evaluation_experiment"]
            ).experiment_id,
            order_by=["attribute.end_time DESC"],
        ),
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


def set_experiments(settings):
    mlflow.set_experiment(settings["defaults"]["agent_experiment"])
    mlflow.set_experiment(settings["defaults"]["dataset_experiment"])
    mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])


def generate_base(dataset_name, settings):

    set_experiments(settings)

    dataset_loader_settings = settings["dataset_loaders"][dataset_name]
    mlflow.set_experiment("dataset")
    with mlflow.start_run() as _:
        log_custom_parameters(
            parameters_normalize(
                constants.DATASET_PARAMETERS_PREFIX,
                dataset_name,
                dataset_loader_settings,
            )
        )
        # client.log_param()
        # for k,v in dataset_loader_settings.items():
        # log_param(k,v)

        from irec.utils.Factory import DatasetLoaderFactory

        dataset_loader_factory = DatasetLoaderFactory()
        dataset_loader = dataset_loader_factory.create(
            dataset_name, dataset_loader_settings
        )
        dataset = dataset_loader.load()

        fname = "./tmp/dataset.pickle"
        log_custom_artifact(fname, dataset)


def download_data(dataset_names):
    import zipfile
    import gdown
    import os

    datasets_ids = {
        "Good Books": "14mT0bNCveFB1wKR_uhG08-GpOd_MxCZ7",
        "Good Reads 10k": "1j3JA8ZUhCAWruvKgSgY1LbYpc0Mdrig8",
        "Kindle 4k": "1CvhZ2GwalzHq9cp5r9SBKliedxfWpBba",
        "Kindle Store": "1JBCBBLDFcY46RKn8vw5u9EeJM_2-oASn",
        "LastFM 5k": "1AcnaOmxJccTaGuxAYH7icv9Vg_yh8r--",
        "MovieLens 1M": "1zQZ3vxEEXFIjpS8mS82B3XSw84SCPe6w",
        "MovieLens 10M": "1pV5PD2Cio41DLGEcN0Fb5MP1cGPzBJBB",
        "MovieLens 20M": "1LOStldkZgOKyaOjd8QhSs8dkM9rPo6dY",
        "MovieLens 25M": "1TMBII4-W_HfXxebwgQCXLZHVDjtX88hr",
        "MovieLens 100k": "1C0lHUQv73v58khSIKBE1VeVSldVsu_gE",
        "Netflix": "13H960S8-I2a-U3V_PmOC1TfOLZrkdm_h",
        "Netflix 10k": "1yynqrIW7GwTGXvuZ0ToPf-SEOGACSil3",
        "Yahoo Music": "1zWxmQ8zJvZQKBgUGK49_O6g6dQ7Mxhzn",
        "Yahoo Music 5k": "1c7HRu7Nlz-gbcc1-HsSTk98PYIZSQWEy",
        "Yahoo Music 10k": "1LMIMFjweVebeewn4D61KX72uIQJlW5d0",
        "Nano Dataset": "1ya8m3dDJ8OzvmDuYPlb6_fgsUrRysfAC",
    }

    dataset_dir = "./data/datasets/"
    url = "https://drive.google.com/uc?id="

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    for dataset in dataset_names:
        print("\nDataset:", dataset)
        output = f"{dataset_dir}{dataset}.zip"
        gdown.download(f"{url}{datasets_ids[dataset]}", output)
        with zipfile.ZipFile(output, "r") as zip_ref:
            zip_ref.extractall(dataset_dir)
        os.remove(output)


def run_agent_with_dataset_parameters(
    agents, dataset_loaders, settings, dataset_agents_parameters, tasks, forced_run
):

    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    if tasks>1:
        executor = ProcessPoolExecutor(max_workers=tasks)
    futures = set()
    # with ProcessPoolExecutor(max_workers=tasks) as executor:
    for dataset_loader_name in dataset_loaders:
        current_settings = settings
        current_settings["defaults"]["dataset_loader"] = dataset_loader_name
        traintest_dataset = load_dataset_experiment(settings)
        for agent_name in agents:
            current_settings["defaults"]["agent"] = agent_name
            current_settings["agents"][agent_name] = dataset_agents_parameters[
                dataset_loader_name
            ][agent_name]
            if tasks>1:
                f = executor.submit(
                    run_agent,
                    traintest_dataset,
                    copy.deepcopy(current_settings),
                    forced_run,
                )
                futures.add(f)
                if len(futures) >= tasks:
                    completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            else:
                run_agent(traintest_dataset,copy.deepcopy(current_settings),forced_run)

    for f in futures:
        f.result()


def print_results_latex_table(
    agents,
    dataset_loaders,
    settings,
    dataset_agents_parameters,
    metrics,
    reference_agent,
    dump,
    type,
):

    from cycler import cycler

    plt.rcParams["axes.prop_cycle"] = cycler(color="krbgmyc")
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["font.size"] = 15
    # metrics_classes = [metrics.Hits, metrics.Recall]
    metrics_classes = [eval("irec.metrics." + i) for i in metrics]

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
    datasets_names = dataset_loaders

    metric_evaluator_name = settings["defaults"]["metric_evaluator"]
    metric_evaluator_parameters = settings["metric_evaluators"][metric_evaluator_name]

    exec(
        f"from irec.metric_evaluators.{metric_evaluator_name} import {metric_evaluator_name}"
    )
    metric_evaluator = eval(metric_evaluator_name)(None, **metric_evaluator_parameters)

    evaluation_policy_name = settings["defaults"]["evaluation_policy"]
    # evaluation_policy_parameters = settings["evaluation_policies"][
    # evaluation_policy_name
    # ]

    exec(
        f"from irec.evaluation_policies.{evaluation_policy_name} import {evaluation_policy_name}"
    )
    # evaluation_policy = eval(evaluation_policy_name)(**evaluation_policy_parameters)

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

    # interactors_classes_names_to_names = {
    # k: v["name"] for k, v in settings["agents_general_settings"].items()
    # }

    print("metric_evaluator_name", metric_evaluator_name)
    if metric_evaluator_name == "StageIterationsMetricEvaluator":
        nums_interactions_to_show = ["1-5", "6-10", "11-15", "16-20", "21-50", "51-100"]
    else:
        nums_interactions_to_show = list(
            map(int, metric_evaluator.interactions_to_evaluate)
        )

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

    datasets_metrics_values = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    datasets_metrics_users_values = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])

    runs_infos = mlflow.list_run_infos(
        mlflow.get_experiment_by_name(
            settings["defaults"]["evaluation_experiment"]
        ).experiment_id,
        order_by=["attribute.end_time DESC"],
    )

    for dataset_loader_name in datasets_names:

        for metric_class_name in metrics_classes_names:
            for agent_name in agents:
                settings["defaults"]["dataset_loader"] = dataset_loader_name
                settings["defaults"]["agent"] = agent_name
                agent_parameters = dataset_agents_parameters[dataset_loader_name][
                    agent_name
                ]
                settings["agents"][agent_name] = agent_parameters
                settings["defaults"]["metric"] = metric_class_name
                # agent = AgentFactory().create(agent_name, agent_parameters)
                # agent_id = get_agent_id(agent_name, agent_parameters)
                # dataset_parameters = settings["dataset_loaders"][dataset_loader_name]
                # metrics_evaluator_name = metric_evaluator.__class__.__name__
                # parameters_agent_run = get_agent_run_parameters(settings)
                parameters_evaluation_run = get_evaluation_run_parameters(settings)

                mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])
                print(dataset_loader_name, agent_name)
                run = already_ran(
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
                # users_items_recommended = metric_values

                datasets_metrics_values[dataset_loader_name][metric_class_name][
                    agent_name
                ].extend(
                    [
                        np.mean(list(metric_values[i].values()))
                        for i in range(len(nums_interactions_to_show))
                    ]
                )
                datasets_metrics_users_values[dataset_loader_name][metric_class_name][
                    agent_name
                ].extend(
                    np.array(
                        [
                            list(metric_values[i].values())
                            for i in range(len(nums_interactions_to_show))
                        ]
                    )
                )

    utility_scores = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: dict()))
    )
    # method_utility_scores = defaultdict(lambda: defaultdict(lambda: dict))
    for num_interaction in range(len(nums_interactions_to_show)):
        for dataset_loader_name in datasets_names:
            for metric_class_name in metrics_classes_names:
                for agent_name in agents:
                    metric_max_value = np.max(
                        list(
                            map(
                                lambda x: x[num_interaction],
                                datasets_metrics_values[dataset_loader_name][
                                    metric_class_name
                                ].values(),
                            )
                        )
                    )
                    metric_min_value = np.min(
                        list(
                            map(
                                lambda x: x[num_interaction],
                                datasets_metrics_values[dataset_loader_name][
                                    metric_class_name
                                ].values(),
                            )
                        )
                    )
                    metric_value = datasets_metrics_values[dataset_loader_name][
                        metric_class_name
                    ][agent_name][num_interaction]
                    print(
                        "metric_value",
                        metric_value,
                        "metric_min_value",
                        metric_min_value,
                    )
                    try:
                        utility_scores[dataset_loader_name][metric_class_name][
                            agent_name
                        ][num_interaction] = (metric_value - metric_min_value) / (
                            metric_max_value - metric_min_value
                        )
                    except Exception as e:
                        print(e)
                        utility_scores[dataset_loader_name][metric_class_name][
                            agent_name
                        ][num_interaction] = 0.0

    for num_interaction in range(len(nums_interactions_to_show)):
        for dataset_loader_name in datasets_names:
            for agent_name in agents:
                us = [
                    utility_scores[dataset_loader_name][metric_class_name][agent_name][
                        num_interaction
                    ]
                    * metrics_weights[metric_class_name]
                    for metric_class_name in metrics_classes_names
                ]
                maut = np.sum(us)
                datasets_metrics_values[dataset_loader_name]["MAUT"][agent_name].append(
                    maut
                )
                datasets_metrics_users_values[dataset_loader_name]["MAUT"][
                    agent_name
                ].append(np.array([maut] * 100))

    if dump:
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

    if type == "pairs":
        pool_of_methods_to_compare = [
            (agents[i], agents[i + 1]) for i in range(0, len(agents) - 1, 2)
        ]
    else:
        pool_of_methods_to_compare = [[agents[i] for i in range(len(agents))]]
    print(pool_of_methods_to_compare)
    for dataset_loader_name in datasets_names:
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
                    datasets_metrics_best[dataset_loader_name][metric_class_name][
                        max(
                            datasets_metrics_values_tmp[dataset_loader_name][
                                metric_class_name
                            ].items(),
                            key=lambda x: x[1][i],
                        )[0]
                    ][i] = True
                    if reference_agent == "lastmethod":
                        best_itr = methods[-1]
                    elif reference_agent is not None:
                        best_itr = reference_agent
                    else:
                        best_itr = max(
                            datasets_metrics_values_tmp[dataset_loader_name][
                                metric_class_name
                            ].items(),
                            key=lambda x: x[1][i],
                        )[0]
                    best_itr_vals = datasets_metrics_values_tmp[dataset_loader_name][
                        metric_class_name
                    ].pop(best_itr)
                    best_itr_val = best_itr_vals[i]
                    second_best_itr = max(
                        datasets_metrics_values_tmp[dataset_loader_name][
                            metric_class_name
                        ].items(),
                        key=lambda x: x[1][i],
                    )[0]
                    second_best_itr_vals = datasets_metrics_values_tmp[
                        dataset_loader_name
                    ][metric_class_name][second_best_itr]
                    second_best_itr_val = second_best_itr_vals[i]
                    # come back with value in dict
                    datasets_metrics_values_tmp[dataset_loader_name][metric_class_name][
                        best_itr
                    ] = best_itr_vals

                    best_itr_users_val = datasets_metrics_users_values[
                        dataset_loader_name
                    ][metric_class_name][best_itr][i]
                    second_best_itr_users_val = datasets_metrics_users_values[
                        dataset_loader_name
                    ][metric_class_name][second_best_itr][i]

                    try:
                        # print(best_itr_users_val)
                        # print(second_best_itr_users_val)
                        statistic, pvalue = scipy.stats.wilcoxon(
                            best_itr_users_val,
                            second_best_itr_users_val,
                        )
                    except Exception as E:
                        print("[ERROR]: Wilcoxon error", E)
                        datasets_metrics_gain[dataset_loader_name][metric_class_name][
                            best_itr
                        ][i] = bullet_str
                        continue

                    if pvalue > 0.05:
                        datasets_metrics_gain[dataset_loader_name][metric_class_name][
                            best_itr
                        ][i] = bullet_str
                    else:
                        # print(best_itr,best_itr_val,second_best_itr,second_best_itr_val,methods)
                        if best_itr_val < second_best_itr_val:
                            datasets_metrics_gain[dataset_loader_name][
                                metric_class_name
                            ][best_itr][i] = triangle_down_str
                        elif best_itr_val > second_best_itr_val:
                            datasets_metrics_gain[dataset_loader_name][
                                metric_class_name
                            ][best_itr][i] = triangle_up_str
                        else:
                            datasets_metrics_gain[dataset_loader_name][
                                metric_class_name
                            ][best_itr][i] = bullet_str

    for metric_name, metric_class_name in zip(metrics_names, metrics_classes_names):
        rtex += generate_metric_interactions_header(
            nums_interactions_to_show, len(datasets_names), metric_name
        )
        for agent_name in agents:
            rtex += "%s & " % (get_agent_pretty_name(agent_name, settings))
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
    create_path_to_file(tex_path)
    open(
        tex_path,
        "w+",
    ).write(res)
    pdf_path = os.path.join(
        settings["defaults"]["data_dir"], settings["defaults"]["pdf_dir"]
    )
    create_path_to_file(pdf_path)
    # print(f"latexmk -pdf -interaction=nonstopmode -output-directory={pdf_path} {tex_path}")
    os.system(
        f'latexmk -pdflatex=pdflatex -pdf -interaction=nonstopmode -output-directory="{pdf_path}" "{tex_path}"'
    )
    useless_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(pdf_path)
        for f in filenames
        if os.path.splitext(f)[1] != ".pdf"
    ]
    for useless_file in useless_files:
        os.remove(useless_file)


def evaluate_agent_with_dataset_parameters(
    agents,
    dataset_loaders,
    settings,
    dataset_agents_parameters,
    metrics,
    tasks,
    forced_run,
):

    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
    from irec.utils.dataset import Dataset

    with ProcessPoolExecutor(max_workers=tasks) as executor:
        futures = set()
        for dataset_loader_name in dataset_loaders:
            settings["defaults"]["dataset_loader"] = dataset_loader_name

            traintest_dataset = load_dataset_experiment(settings)

            data = np.vstack(
                (traintest_dataset.train.data, traintest_dataset.test.data)
            )

            dataset = Dataset(data)
            dataset.set_parameters()
            
            for agent_name in agents:
                settings["defaults"]["agent"] = agent_name
                settings["agents"][agent_name] = dataset_agents_parameters[
                    dataset_loader_name
                ][agent_name]

                for metric_name in metrics:
                    settings["defaults"]["metric"] = metric_name
                    f = executor.submit(
                        evaluate_itr, dataset, copy.deepcopy(settings), forced_run
                    )
                    futures.add(f)
                    if len(futures) >= tasks:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
        for f in futures:
            f.result()


def run_agent_search(
    agents, dataset_loaders, settings, agents_search_parameters, tasks, forced_run
):

    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    with ProcessPoolExecutor(max_workers=tasks) as executor:
        futures = set()
        for dataset_loader_name in dataset_loaders:
            settings["defaults"]["dataset_loader"] = dataset_loader_name
            data = load_dataset_experiment(settings)
            for agent_name in agents:
                settings["defaults"]["agent"] = agent_name
                for agent_og_parameters in agents_search_parameters[agent_name]:
                    settings["agents"][agent_name] = agent_og_parameters
                    f = executor.submit(
                        run_agent, data, copy.deepcopy(settings), forced_run
                    )
                    futures.add(f)
                    if len(futures) >= tasks:
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
        for f in futures:
            f.result()


def eval_agent_search(
    agents, dataset_loaders, settings, agents_search_parameters, metrics, tasks
):

    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    with ProcessPoolExecutor(max_workers=tasks) as executor:
        futures = set()
        for dataset_loader_name in dataset_loaders:
            settings["defaults"]["dataset_loader"] = dataset_loader_name
            traintest = load_dataset_experiment(settings)

            data = np.vstack((traintest.train.data, traintest.test.data))
            dataset = copy.copy(traintest.train)
            dataset.data = data
            dataset.set_parameters()
            for agent_name in agents:
                settings["defaults"]["agent"] = agent_name
                for agent_og_parameters in agents_search_parameters[agent_name]:
                    settings["agents"][agent_name] = agent_og_parameters
                    for metric_name in metrics:
                        settings["defaults"]["metric"] = metric_name
                        f = executor.submit(
                            evaluate_itr, dataset, copy.deepcopy(settings), False
                        )
                        futures.add(f)
                        if len(futures) >= tasks:
                            completed, futures = wait(
                                futures, return_when=FIRST_COMPLETED
                            )
        for f in futures:
            f.result()


def print_agent_search(
    agents,
    dataset_loaders,
    settings,
    dataset_agents_parameters,
    agents_search_parameters,
    metrics,
    dump,
    top_save,
):

    # import irec.metrics
    # import irec.evaluation_policies

    # evaluation_policy_name = settings["defaults"]["evaluation_policy"]
    # evaluation_policy_parameters = settings["evaluation_policies"][
    # evaluation_policy_name
    # ]
    # metrics_classes = [irec.metrics.Hits]
    # metrics_names = ["Cumulative Hits"]
    # evaluation_policy = eval("irec.evaluation_policies." + evaluation_policy_name)(
    # **evaluation_policy_parameters
    # )

    # interactors_classes_names_to_names = {
    # k: v["name"] for k, v in settings["agents_general_settings"].items()
    # }

    # metric_evaluator_parameters = settings["metric_evaluators"][
    # settings["defaults"]["metric_evaluator"]
    # ]

    # metric_class = eval("irec.metrics." + settings["defaults"]["metric"])

    # metric_evaluator = eval(
    # "irec.metric_evaluators." + settings["defaults"]["metric_evaluator"]
    # )(None, **metric_evaluator_parameters)

    datasets_metrics_values = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    )

    for dataset_loader_name in dataset_loaders:
        settings["defaults"]["dataset_loader"] = dataset_loader_name
        # traintest_dataset = load_dataset_experiment(settings)

        for metric_name in metrics:
            for agent_name in agents:
                for agent_parameters in agents_search_parameters[agent_name]:
                    print(agent_name)
                    settings["defaults"]["metric"] = metric_name
                    settings["defaults"]["agent"] = agent_name
                    settings["agents"][agent_name] = agent_parameters
                    # agent = create_agent(agent_name, agent_parameters)
                    agent = AgentFactory().create(agent_name, agent_parameters)
                    # agent_id = get_agent_id(agent_name, agent_parameters)
                    try:
                        metric_values = load_evaluation_experiment(settings)
                        if metric_values is None:
                            continue

                    except errors.EvaluationRunNotFoundError as e:
                        print(e)
                        continue
                    except EOFError as e:
                        print(e)
                        continue

                    datasets_metrics_values[settings["defaults"]["dataset_loader"]][
                        settings["defaults"]["metric"]
                    ][agent.name][json.dumps(agent_parameters)] = metric_values[-1]
                    print(metric_values[-1])
    # ','.join(map(lambda x: str(x[0])+'='+str(x[1]),list(parameters.items())))

    # print(datasets_metrics_values)
    for k1, v1 in datasets_metrics_values.items():
        for _, v2 in v1.items():
            for k3, v3 in v2.items():
                values = np.array(list(v3.values()))
                keys = list(v3.keys())
                idxs = np.argsort(values)[::-1]
                keys = [keys[i] for i in idxs]
                values = [values[i] for i in idxs]
                if dump:
                    if k1 not in dataset_agents_parameters:
                        dataset_agents_parameters[k1] = {}
                    dataset_agents_parameters[k1][k3] = json.loads(keys[0])
                if top_save:
                    print(f"{k3}:")
                    # print('\tparameters:')
                    agent_parameters, _ = json.loads(keys[0]), values[0]
                    for name, value in agent_parameters.items():
                        print(f"\t\t{name}: {value}")
                else:
                    for k4, v4 in zip(keys, values):
                        k4 = yaml.safe_load(k4)
                        # k4 = ','.join(map(lambda x: str(x[0])+'='+str(x[1]),list(k4.items())))
                        print(f"{k3}({k4}) {v4:.5f}")

    if dump:
        print("Saved parameters!")
        open("settings" + sep + "dataset_agents.yaml", "w").write(
            yaml.dump(dataset_agents_parameters)
        )


def print_results_latex_horizontal_table(
    agents,
    dataset_loader,
    settings,
    dataset_agents_parameters,
    metrics,
    reference_agent,
    dump,
    type,
):

    dataset_loaders = [dataset_loader]
    from cycler import cycler

    plt.rcParams["axes.prop_cycle"] = cycler(color="krbgmyc")
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["font.size"] = 15
    metrics_classes = [eval("irec.metrics." + i) for i in metrics]

    metrics_classes_names = list(map(lambda x: x.__name__, metrics_classes))
    metrics_names = metrics_classes_names
    datasets_names = dataset_loaders

    metric_evaluator_name = settings["defaults"]["metric_evaluator"]
    metric_evaluator_parameters = settings["metric_evaluators"][metric_evaluator_name]

    exec(
        f"from irec.metric_evaluators.{metric_evaluator_name} import {metric_evaluator_name}"
    )
    metric_evaluator = eval(metric_evaluator_name)(None, **metric_evaluator_parameters)

    evaluation_policy_name = settings["defaults"]["evaluation_policy"]

    exec(
        f"from irec.evaluation_policies.{evaluation_policy_name} import {evaluation_policy_name}"
    )
    metrics_weights = {i: 1 / len(metrics_classes_names) for i in metrics_classes_names}

    print("metric_evaluator_name", metric_evaluator_name)
    if metric_evaluator_name == "StageIterationsMetricEvaluator":
        nums_interactions_to_show = ["1-5", "6-10", "11-15", "16-20", "21-50", "51-100"]
    else:
        nums_interactions_to_show = list(
            map(int, metric_evaluator.interactions_to_evaluate)
        )

    def generate_table_spec():
        res = "|"
        for i in range(1*len(metrics_names) + len(nums_interactions_to_show) * len(metrics_names)):
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
    Dataset & %s \\\hline
    """ % (
        generate_table_spec(),
        r"\multicolumn{%d}{c|}{%s}" % (len(nums_interactions_to_show)*len(metrics_names), datasets_names[0])
    )
    rtex_footer = r"""
    \end{tabular}
    \end{document}
    """
    rtex = ""

    datasets_metrics_values = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    datasets_metrics_users_values = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])

    runs_infos = mlflow.list_run_infos(
        mlflow.get_experiment_by_name(
            settings["defaults"]["evaluation_experiment"]
        ).experiment_id,
        order_by=["attribute.end_time DESC"],
    )

    for dataset_loader_name in datasets_names:

        for metric_class_name in metrics_classes_names:
            for agent_name in agents:
                settings["defaults"]["dataset_loader"] = dataset_loader_name
                settings["defaults"]["agent"] = agent_name
                agent_parameters = dataset_agents_parameters[dataset_loader_name][
                    agent_name
                ]
                settings["agents"][agent_name] = agent_parameters
                settings["defaults"]["metric"] = metric_class_name
                parameters_evaluation_run = get_evaluation_run_parameters(settings)

                mlflow.set_experiment(settings["defaults"]["evaluation_experiment"])
                print(dataset_loader_name, agent_name)
                run = already_ran(
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
                # users_items_recommended = metric_values

                datasets_metrics_values[dataset_loader_name][metric_class_name][
                    agent_name
                ].extend(
                    [
                        np.mean(list(metric_values[i].values()))
                        for i in range(len(nums_interactions_to_show))
                    ]
                )
                datasets_metrics_users_values[dataset_loader_name][metric_class_name][
                    agent_name
                ].extend(
                    np.array(
                        [
                            list(metric_values[i].values())
                            for i in range(len(nums_interactions_to_show))
                        ]
                    )
                )

    # utility_scores = defaultdict(
        # lambda: defaultdict(lambda: defaultdict(lambda: dict()))
    # )
    # for num_interaction in range(len(nums_interactions_to_show)):
        # for dataset_loader_name in datasets_names:
            # for metric_class_name in metrics_classes_names:
                # for agent_name in agents:
                    # metric_max_value = np.max(
                        # list(
                            # map(
                                # lambda x: x[num_interaction],
                                # datasets_metrics_values[dataset_loader_name][
                                    # metric_class_name
                                # ].values(),
                            # )
                        # )
                    # )
                    # metric_min_value = np.min(
                        # list(
                            # map(
                                # lambda x: x[num_interaction],
                                # datasets_metrics_values[dataset_loader_name][
                                    # metric_class_name
                                # ].values(),
                            # )
                        # )
                    # )
                    # metric_value = datasets_metrics_values[dataset_loader_name][
                        # metric_class_name
                    # ][agent_name][num_interaction]
                    # print(
                        # "metric_value",
                        # metric_value,
                        # "metric_min_value",
                        # metric_min_value,
                    # )
                    # try:
                        # utility_scores[dataset_loader_name][metric_class_name][
                            # agent_name
                        # ][num_interaction] = (metric_value - metric_min_value) / (
                            # metric_max_value - metric_min_value
                        # )
                    # except Exception as e:
                        # print(e)
                        # utility_scores[dataset_loader_name][metric_class_name][
                            # agent_name
                        # ][num_interaction] = 0.0

    # for num_interaction in range(len(nums_interactions_to_show)):
        # for dataset_loader_name in datasets_names:
            # for agent_name in agents:
                # us = [
                    # utility_scores[dataset_loader_name][metric_class_name][agent_name][
                        # num_interaction
                    # ]
                    # * metrics_weights[metric_class_name]
                    # for metric_class_name in metrics_classes_names
                # ]
                # maut = np.sum(us)
                # datasets_metrics_values[dataset_loader_name]["MAUT"][agent_name].append(
                    # maut
                # )
                # datasets_metrics_users_values[dataset_loader_name]["MAUT"][
                    # agent_name
                # ].append(np.array([maut] * 100))

    # if dump:
        # with open("datasets_metrics_values.pickle", "wb") as f:
            # pickle.dump(json.loads(json.dumps(datasets_metrics_values)), f)

    # metrics_classes_names.append("MAUT")
    # metrics_names.append("MAUT")

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

    if type == "pairs":
        pool_of_methods_to_compare = [
            (agents[i], agents[i + 1]) for i in range(0, len(agents) - 1, 2)
        ]
    else:
        pool_of_methods_to_compare = [[agents[i] for i in range(len(agents))]]
    print(pool_of_methods_to_compare)
    for dataset_loader_name in datasets_names:
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
                    datasets_metrics_best[dataset_loader_name][metric_class_name][
                        max(
                            datasets_metrics_values_tmp[dataset_loader_name][
                                metric_class_name
                            ].items(),
                            key=lambda x: x[1][i],
                        )[0]
                    ][i] = True
                    if reference_agent == "lastmethod":
                        best_itr = methods[-1]
                    elif reference_agent is not None:
                        best_itr = reference_agent
                    else:
                        best_itr = max(
                            datasets_metrics_values_tmp[dataset_loader_name][
                                metric_class_name
                            ].items(),
                            key=lambda x: x[1][i],
                        )[0]
                    best_itr_vals = datasets_metrics_values_tmp[dataset_loader_name][
                        metric_class_name
                    ].pop(best_itr)
                    best_itr_val = best_itr_vals[i]
                    second_best_itr = max(
                        datasets_metrics_values_tmp[dataset_loader_name][
                            metric_class_name
                        ].items(),
                        key=lambda x: x[1][i],
                    )[0]
                    second_best_itr_vals = datasets_metrics_values_tmp[
                        dataset_loader_name
                    ][metric_class_name][second_best_itr]
                    second_best_itr_val = second_best_itr_vals[i]
                    # come back with value in dict
                    datasets_metrics_values_tmp[dataset_loader_name][metric_class_name][
                        best_itr
                    ] = best_itr_vals

                    best_itr_users_val = datasets_metrics_users_values[
                        dataset_loader_name
                    ][metric_class_name][best_itr][i]
                    second_best_itr_users_val = datasets_metrics_users_values[
                        dataset_loader_name
                    ][metric_class_name][second_best_itr][i]

                    try:
                        # print(best_itr_users_val)
                        # print(second_best_itr_users_val)
                        statistic, pvalue = scipy.stats.wilcoxon(
                            best_itr_users_val,
                            second_best_itr_users_val,
                        )
                    except Exception as E:
                        print("[ERROR]: Wilcoxon error", E)
                        datasets_metrics_gain[dataset_loader_name][metric_class_name][
                            best_itr
                        ][i] = bullet_str
                        continue

                    if pvalue > 0.05:
                        datasets_metrics_gain[dataset_loader_name][metric_class_name][
                            best_itr
                        ][i] = bullet_str
                    else:
                        # print(best_itr,best_itr_val,second_best_itr,second_best_itr_val,methods)
                        if best_itr_val < second_best_itr_val:
                            datasets_metrics_gain[dataset_loader_name][
                                metric_class_name
                            ][best_itr][i] = triangle_down_str
                        elif best_itr_val > second_best_itr_val:
                            datasets_metrics_gain[dataset_loader_name][
                                metric_class_name
                            ][best_itr][i] = triangle_up_str
                        else:
                            datasets_metrics_gain[dataset_loader_name][
                                metric_class_name
                            ][best_itr][i] = bullet_str


    rtex+= "\\rowcolor{Gray} Measure & "
    rtex+= "&".join([
                r"\multicolumn{%d}{c|}{%s}" % (len(nums_interactions_to_show), i)
                for i in metrics_names
        ]) + '\\\\\\hline\n'

    rtex+= "\\rowcolor{Gray} T"
    for _ in metrics_names:
        rtex += " & "
        rtex += "&".join([
                    r"%d" % (i)
                    for i in nums_interactions_to_show
            ])
    rtex += '\\\\\\hline\\hline\n'

    for dataset_name in datasets_names:
        for agent_name in agents:
            rtex += "%s" % (get_agent_pretty_name(agent_name, settings))
            for metric_name, metric_class_name in zip(metrics_names, metrics_classes_names):
                rtex += " & "
                rtex += " & ".join(
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
            rtex += r"\\\hline" + "\n"

    res = rtex_header + rtex + rtex_footer

    tmp = "_".join([dataset_name for dataset_name in datasets_names])
    tex_path = os.path.join(
        settings["defaults"]["data_dir"],
        settings["defaults"]["tex_dir"],
        f"tableh_{tmp}.tex",
    )
    create_path_to_file(tex_path)
    open(
        tex_path,
        "w+",
    ).write(res)
    pdf_path = os.path.join(
        settings["defaults"]["data_dir"], settings["defaults"]["pdf_dir"]
    )
    create_path_to_file(pdf_path)
    # print(f"latexmk -pdf -interaction=nonstopmode -output-directory={pdf_path} {tex_path}")
    os.system(
        f'latexmk -pdflatex=pdflatex -pdf -interaction=nonstopmode -output-directory="{pdf_path}" "{tex_path}"'
    )
    useless_files = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(pdf_path)
        for f in filenames
        if os.path.splitext(f)[1] != ".pdf"
    ]
    for useless_file in useless_files:
        os.remove(useless_file)
