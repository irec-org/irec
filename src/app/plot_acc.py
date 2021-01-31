from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

import inquirer
import interactors
import mf
from utils.InteractorRunner import InteractorRunner
from sklearn.decomposition import NMF
import numpy as np
import scipy.sparse
from utils.DatasetManager import DatasetManager
import yaml
from metric import InteractionMetricsEvaluator, CumulativeMetricsEvaluator
from utils.dataset import Dataset
from utils.PersistentDataManager import PersistentDataManager
from utils.InteractorCache import InteractorCache
import metric
import matplotlib.pyplot as plt
from utils.DirectoryDependent import DirectoryDependent
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='krbgmyc')
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 15
BUFFER_SIZE_EVALUATOR = 50

metrics_classes = [metric.Precision, metric.Recall, metric.Hits]

dm = DatasetManager()
dataset_preprocessor = dm.request_dataset_preprocessor()
dm.initialize_engines(dataset_preprocessor)
dm.load()

interactors_preprocessor_paramaters = yaml.load(
    open("settings" + sep + "interactors_preprocessor_parameters.yaml"),
    Loader=yaml.SafeLoader)
interactors_general_settings = yaml.load(
    open("settings" + sep + "interactors_general_settings.yaml"),
    Loader=yaml.SafeLoader)

evaluation_policies_parameters = yaml.load(
    open("settings" + sep + "evaluation_policies_parameters.yaml"),
    Loader=yaml.SafeLoader)

interactors_classes_names_to_names = {
    k: v['name'] for k, v in interactors_general_settings.items()
}

ir = InteractorRunner(dm, interactors_general_settings,
                      interactors_preprocessor_paramaters,
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

fig, axs = plt.subplots(nrows=len(metrics_evaluators), ncols=len(metrics_classes), figsize=(18, 17))
fig.suptitle(
    f"Top-{evaluation_policy.interaction_size} recommendation, {dm.dataset_preprocessor.name}"
)

i = 0
for metric_evaluator in metrics_evaluators:
    j = 0
    for metric_name in map(lambda x: x.__name__, metrics_classes):
        ax = axs[i, j]
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

        if isinstance(metric_evaluator, InteractionMetricsEvaluator):
            ax.set_xlabel("Interactions", size=18)
        elif isinstance(metric_evaluator, CumulativeMetricsEvaluator):
            ax.set_xlabel("Time", size=18)
        ax.set_ylabel(metric_name, size=18)
        ax.yaxis.set_label_coords(-0.1,1.02)
        j += 1
    i += 1

s = fig.subplotpars

fig.legend([
    ir.get_interactor_name(itr_class.__name__)
    for itr_class in interactors_classes
],
           loc='lower center',
           bbox_to_anchor=[s.left, s.top + 0.04, s.right - s.left, 0.05],
           ncol=6,
           mode="expand",
           borderaxespad=0,
           bbox_transform=fig.transFigure,
           fancybox=False,
           edgecolor="k")
fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],
                         f'plot_{dm.dataset_preprocessor.name}.png'),
            bbox_inches='tight')
