from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import value_functions
import mf
from lib.utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from lib.utils.DatasetManager import DatasetManager
import yaml
from metrics import InteractionMetricsEvaluator, CumulativeMetricsEvaluator
from lib.utils.dataset import Dataset
from lib.utils.PersistentDataManager import PersistentDataManager
from lib.utils.InteractorCache import InteractorCache
import metrics
import matplotlib.pyplot as plt
from lib.utils.DirectoryDependent import DirectoryDependent
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='krbgmyc')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15

BUFFER_SIZE_EVALUATOR = 50

metrics_classes = [metrics.Precision, metrics.Recall, metrics.Hits]

dm = DatasetManager()
dataset_preprocessor = dm.request_dataset_preprocessor()
dm.initialize_engines(dataset_preprocessor)
dm.load()

interactors_preprocessor_parameters = yaml.load(
    open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
    Loader=yaml.SafeLoader)
interactors_general_settings = yaml.load(
    open("settings" + sep + "interactors_general_settings.yaml"),
    Loader=yaml.SafeLoader)

evaluation_policies_parameters = yaml.load(
    open("settings" + sep + "evaluation_policies_parameters.yaml"),
    Loader=yaml.SafeLoader)

interactors_classes_names_to_names = {
    k: v['name']
    for k, v in interactors_general_settings.items()
}

ir = InteractorRunner(dm, interactors_general_settings,
                      interactors_preprocessor_parameters,
                      evaluation_policies_parameters)
interactors_classes = ir.select_interactors()

data = np.vstack(
    (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))

dataset = Dataset(data)
dataset.update_from_data()
dataset.update_num_total_users_items()

metrics_evaluators = [
    InteractionMetricsEvaluator(dataset, metrics_classes),
    CumulativeMetricsEvaluator(BUFFER_SIZE_EVALUATOR, dataset, metrics_classes)
]

evaluation_policy = ir.get_interactors_evaluation_policy()

for metric_evaluator in metrics_evaluators:
    for metric_name in map(lambda x: x.__name__, metrics_classes):
        fig, ax = plt.subplots()
        for itr_class in interactors_classes:
            itr = ir.create_interactor(itr_class)
            pdm = PersistentDataManager(directory='results')

            metrics_pdm = PersistentDataManager(directory='metrics')
            metric_values = metrics_pdm.load(
                os.path.join(
                    InteractorCache().get_id(dm, evaluation_policy, itr),
                    metric_evaluator.get_id(), metric_name))
            if isinstance(metric_evaluator, InteractionMetricsEvaluator):
                ax.plot(range(1,
                              len(metric_values) + 1),
                        metric_values,
                        label=interactors_classes_names_to_names[
                            itr_class.__name__])
            elif isinstance(metric_evaluator, CumulativeMetricsEvaluator):
                ax.plot(range(1,
                              len(metric_values) * BUFFER_SIZE_EVALUATOR,
                              BUFFER_SIZE_EVALUATOR),
                        metric_values,
                        label=interactors_classes_names_to_names[
                            itr_class.__name__])

        plt.legend(bbox_to_anchor=(0 - 0.1, 1.1, 1 + 2 * 0.1, 0.2),
                   loc="lower left",
                   mode="expand",
                   borderaxespad=0,
                   ncol=3,
                   fontsize=12)

        if isinstance(metric_evaluator, InteractionMetricsEvaluator):
            ax.set_xlabel("Interactions", size=18)
        elif isinstance(metric_evaluator, CumulativeMetricsEvaluator):
            ax.set_xlabel("Time", size=18)
        ax.set_title(
            f"Top-{evaluation_policy.interaction_size} recommendation",
            size=18)
        ax.set_ylabel(metric_name, size=18)
        # ax.set_xticks(list(range(1,evaluation_policy.num_interactions+1,evaluation_policy.num_interactions//4)) + [evaluation_policy.num_interactions])
        # for tick in ax.xaxis.get_major_ticks()+ax.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(14)

        fig.savefig(os.path.join(
            DirectoryDependent().DIRS["img"],
            f'plot_{metric_evaluator.get_id()}_{dm.dataset_preprocessor.name}_{metric_name}.png'
        ),
                    bbox_inches='tight')
