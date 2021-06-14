from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")

from lib.utils.DirectoryDependent import DirectoryDependent
from lib.utils.DatasetManager import DatasetManager
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import scipy.stats
import pandas as pd
dm = DatasetManager()
dataset_preprocessor = dm.request_dataset_preprocessor()
dm.initialize_engines(dataset_preprocessor)
dm.load()
data = np.vstack(
    (dm.dataset_preprocessed[0].data, dm.dataset_preprocessed[1].data))

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# self.var = np.var(training_matrix)
# self.user_var = np.mean(np.var(training_matrix,axis=1))
# self.item_var = np.mean(np.var(training_matrix,axis=0))
df = pd.DataFrame(dm.train_dataset.data)
var = df[2].var()
user_var = df.groupby(0).var()[2].mean()
item_var = df.groupby(1).var()[2].mean()
print('var', var)
print('user_var', user_var)
print('item_var', item_var)
print('user_lambda', var / user_var)
print(dm.dataset_preprocessed[0].num_users)
print(dm.dataset_preprocessed[0].num_items)
del df

raise SystemExit
print(scipy.stats.describe(data))
lowest_value = 0

for ax, matrix, name in zip(axs[0, :], [
        scipy.sparse.csr_matrix(
            (dm.test_dataset.data[:, 2],
             (dm.test_dataset.data[:, 0], dm.test_dataset.data[:, 1])),
            (dm.test_dataset.num_total_users,
             dm.test_dataset.num_total_items)),
        scipy.sparse.csr_matrix(
            (dm.train_dataset.data[:, 2],
             (dm.train_dataset.data[:, 0], dm.train_dataset.data[:, 1])),
            (dm.train_dataset.num_total_users,
             dm.train_dataset.num_total_items)),
], ['Test', 'Train']):
    # dont work with sparse matrix
    # dataset = np.count_nonzero(matrix,axis=1)
    dataset = np.sum(matrix > lowest_value, axis=1).A.flatten()
    dataset = dataset[dataset > 0]
    ax.hist(dataset, color='k', bins=100)
    ax.set_xlabel('#Consumption')
    ax.set_ylabel('#Users')
    ax.set_title(f'{name} users')
    output_format = '#Users %d\nMean($\\bar{x}$) %.2f\nMedian($\\tilde{x}$) %d\nMin %d Max %d\nStd(s) %.2f'
    output_format += '''
Percentile$_{5}$ %d
Percentile$_{10}$ %d
Percentile$_{30}$ %d
Percentile$_{50}$ %d
Percentile$_{70}$ %d
Percentile$_{90}$ %d'''
    ax.annotate(output_format % (
        dataset.shape[0],
        np.mean(dataset),
        np.median(dataset),
        np.min(dataset),
        np.max(dataset),
        np.std(dataset),
        np.percentile(dataset, 5),
        np.percentile(dataset, 10),
        np.percentile(dataset, 30),
        np.percentile(dataset, 50),
        np.percentile(dataset, 70),
        np.percentile(dataset, 90),
    ),
                xy=(0.37, 0.22),
                xycoords='axes fraction',
                fontsize=14,
                bbox=dict(boxstyle="square", fc="w"))

consumption_matrix = scipy.sparse.csr_matrix(
    (data[:, 2], (data[:, 0], data[:, 1])),
    (dm.test_dataset.num_total_users, dm.test_dataset.num_total_items))

consumption_time_matrix = scipy.sparse.csr_matrix(
    (data[:, 3], (data[:, 0], data[:, 1])),
    (dm.test_dataset.num_total_users, dm.test_dataset.num_total_items))

users_start_time = []
for uid in range(consumption_time_matrix.shape[0]):
    users_start_time.append(np.min(consumption_time_matrix[uid].data))

users_start_time = np.array(users_start_time)
# users_start_time = np.where(consumption_time_matrix.A > 0,consumption_time_matrix.A,np.inf).min(axis=1)

users_by_time = np.argsort(users_start_time)
users_start_datetime = [
    datetime.fromtimestamp(i) for i in users_start_time[users_by_time]
]
axs[1, 0].plot(users_start_datetime, color='k', linewidth=2)
axs[1, 0].set_xlabel('Users')
axs[1, 0].set_ylabel('First rating')

users_num_consumption = np.sum(consumption_matrix > lowest_value,
                               axis=1)[users_by_time].A.flatten()
hist_values = np.sum(matrix > lowest_value, axis=1).A.flatten()
hist_values = hist_values[hist_values > 0]

axs[1, 1].hist(hist_values, color='k', bins=100)
ax.set_xlabel('#Consumption')
ax.set_ylabel('#Users')

axs[1, 1].annotate(
    'Pearson(time,#consumption) %.2f' %
    (scipy.stats.pearsonr(users_by_time, users_num_consumption)[0]),
    xy=(0.02, 0.9),
    xycoords='axes fraction',
    fontsize=14,
    bbox=dict(boxstyle="square", fc="w"))

num_consumption = np.sum(consumption_matrix > lowest_value)

plt.annotate('Sparsity {:.2f}%, #Users {}, #Items {}, #Consumption {}'.format(
    100 * (1 - num_consumption / np.prod(consumption_matrix.shape)),
    consumption_matrix.shape[0],
    consumption_matrix.shape[1],
    num_consumption,
),
             xy=(0.5, 0.94),
             xycoords='figure fraction',
             fontsize=14,
             bbox=dict(boxstyle="square", fc="w"),
             ha='center')
fig.suptitle(dm.dataset_preprocessor.name, fontsize=14)
fig.savefig(os.path.join(DirectoryDependent().DIRS["img"],
                         f'{dm.dataset_preprocessor.name}_comsumption.png'),
            bbox_inches='tight')
