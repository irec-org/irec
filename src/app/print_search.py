from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir)

import json
import inquirer
import lib.value_functions
import lib.mf
import utils
import lib.evaluation_policies
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from metrics import CumulativeInteractionMetricsEvaluator
from lib.utils.dataset import Dataset
from lib.utils.PersistentDataManager import PersistentDataManager
import metrics
import matplotlib.pyplot as plt
from lib.utils.DirectoryDependent import DirectoryDependent
from cycler import cycler
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='Grid search')
parser.add_argument('-t', default=False, action='store_true',help='Print only top 1')
parser.add_argument('-d', default=False, action='store_true',help='Save best')
parser.add_argument('-m',nargs='*')
parser.add_argument('-b',nargs='*')
settings = utils.load_settings()
utils.load_settings_to_parser(settings,parser)
args = parser.parse_args()
settings = utils.sync_settings_from_args(settings,args)

# print(args.b,args.m)

metrics_classes = [metrics.Hits]
metrics_names = ['Cumulative Hits']

dm = DatasetManager()
# datasets_preprocessors = dm.request_datasets_preprocessors()


evaluation_policy_name = settings['defaults']['interactors_evaluation_policy']
evaluation_policy_parameters = settings['evaluation_policies_parameters'][evaluation_policy_name]
evaluation_policy=eval('lib.evaluation_policies.'+evaluation_policy_name)(**evaluation_policy_parameters)


interactors_classes_names_to_names = {
    k: v['name'] for k, v in settings['interactors_general_settings'].items()
}

# interactors_classes = ir.select_interactors()
# interactors_classes = [eval('lib.value_functions.'+interactor) for interactor in args.m]
datasets_preprocessors = [settings['datasets_preprocessors_parameters'][base] for base in args.b]

metrics_evaluator = CumulativeInteractionMetricsEvaluator(None, metrics_classes)

datasets_metrics_values = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
print(datasets_preprocessors)

for dataset_preprocessor in datasets_preprocessors:
    dm.initialize_engines(dataset_preprocessor)

    for metric_class_name in map(lambda x: x.__name__, metrics_classes):
        for agent_name in args.m:
            for parameters in settings['agents_search_parameters'][agent_name]:
                agent = utils.create_agent(agent_name,parameters)
                agent_id = utils.get_agent_id(agent_name,parameters)
                pdm = PersistentDataManager(directory='results')

                metrics_pdm = PersistentDataManager(directory='metrics')
                try:
                    metric_values = metrics_pdm.load(
                        os.path.join(
                            utils.get_experiment_run_id(
                dm, evaluation_policy, agent_id),
                            metrics_evaluator.get_id(), metric_class_name))
                    datasets_metrics_values[dataset_preprocessor['name']][
                            metric_class_name][agent.name][json.dumps(parameters)] = metric_values[-1]
                except:
                    print(f"Could not load metric {dataset_preprocessor['name']} {itr_class.__name__}")
                    pass
# ','.join(map(lambda x: str(x[0])+'='+str(x[1]),list(parameters.items())))

# print(datasets_metrics_values)
for k1, v1 in datasets_metrics_values.items():
    for k2, v2 in v1.items():
        for k3, v3 in v2.items():
            values = np.array(list(v3.values()))
            keys = list(v3.keys())
            idxs = np.argsort(values)[::-1]
            keys = [keys[i] for i in idxs]
            values = [values[i] for i in idxs]
            if args.d:
                settings['agents_preprocessor_parameters'][k1][k3] = json.loads(keys[0])
            if args.t:
                print(f"{k3}:")
                # print('\tparameters:')
                parameters, metric_value = json.loads(keys[0]),values[0]
                for name, value in parameters.items():
                    print(f'\t\t{name}: {value}')
            else:
                for k4, v4 in zip(keys,values):
                    k4 = yaml.safe_load(k4)
                    # k4 = ','.join(map(lambda x: str(x[0])+'='+str(x[1]),list(k4.items())))
                    print(f"{k3}({k4}) {v4:.5f}")

if args.d:
    print("Saved parameters!")
    open("settings" + sep + "agents_preprocessor_parameters.yaml",'w').write(yaml.dump(utils.default_to_regular(settings['agents_preprocessor_parameters'])))
