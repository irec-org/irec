from util import DatasetFormatter
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import scipy.stats
dsf = DatasetFormatter()
dsf = dsf.load()

fig, axs = plt.subplots(nrows=2,ncols=2,figsize=(10,10))

lowest_value = np.min(dsf.consumption_matrix)

print(scipy.stats.describe(dsf.consumption_matrix.data))
# raise SystemExit

for ax, matrix, name in zip(axs[0,:],
                            [dsf.consumption_matrix[dsf.test_uids],dsf.consumption_matrix[dsf.train_uids]],
                            ['Test','Train']):
    # dont work with sparse matrix
    # dataset = np.count_nonzero(matrix,axis=1)
    dataset = np.sum(matrix>lowest_value,axis=1).A.flatten()
    ax.hist(dataset,color='k',bins=100)
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
    ax.annotate(output_format%(
        dataset.shape[0],
        np.mean(dataset),
        np.median(dataset),
        np.min(dataset),np.max(dataset),
        np.std(dataset),
        np.percentile(dataset,5),
        np.percentile(dataset,10),
        np.percentile(dataset,30),
        np.percentile(dataset,50),
        np.percentile(dataset,70),
        np.percentile(dataset,90),
    ),
                xy=(0.37,0.22),xycoords='axes fraction',fontsize=14,
                bbox=dict(boxstyle="square", fc="w"))



users_by_time = np.argsort(dsf.users_start_time)
users_start_datetime=[datetime.fromtimestamp(i) for i in dsf.users_start_time[users_by_time]]
axs[1,0].plot(users_start_datetime,color='k',linewidth=2)
axs[1,0].set_xlabel('Users')
axs[1,0].set_ylabel('First rating')

# dont work with sparse matrix
# users_num_consumption = np.count_nonzero(dsf.consumption_matrix,axis=1)[users_by_time]
users_num_consumption = np.sum(dsf.consumption_matrix>lowest_value,axis=1)[users_by_time].A.flatten()
# print(users_num_consumption)
# print(np.argmin(users_num_consumption))
# print(np.min(users_num_consumption))
# print(len(users_by_time))
# print(dsf.consumption_matrix.shape[0])
# print(len(users_num_consumption))

axs[1,1].bar(x=list(range(dsf.consumption_matrix.shape[0])),height=users_num_consumption,color='k',linewidth=2)
axs[1,1].set_xlabel('Users')
axs[1,1].set_ylabel('#Consumption')

axs[1,1].annotate('Pearson(time,#consumption) %.2f'%(scipy.stats.pearsonr(users_by_time,users_num_consumption)[0]),xy=(0.02,0.9),xycoords='axes fraction',fontsize=14,
                  bbox=dict(boxstyle="square", fc="w"))
# dont work with sparse matrix
# num_consumption = np.count_nonzero(dsf.consumption_matrix)
num_consumption = np.sum(dsf.consumption_matrix>lowest_value)

plt.annotate('Sparsity {:.2f}%, #Users {}, #Items {}, #Consumption {}'.format(
    100*(1-num_consumption/np.prod(dsf.consumption_matrix.shape)),
    dsf.consumption_matrix.shape[0],
    dsf.consumption_matrix.shape[1],
    num_consumption,
),
             xy=(0.5,0.94),xycoords='figure fraction',fontsize=14,
                  bbox=dict(boxstyle="square", fc="w"),ha='center')
fig.suptitle(dsf.PRETTY[dsf.base],fontsize=14)
fig.savefig(f'img/{dsf.base}_comsumption.png',bbox_inches = 'tight')
