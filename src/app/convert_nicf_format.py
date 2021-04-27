import os
from collections import defaultdict

import numpy as np
data = np.loadtxt('../../data/datasets/ml-1m/ratings.dat',
        delimiter='::')

uids_consumes = defaultdict(list)
for i in range(len(data)):
    uids_consumes[int(data[i,0])].append((int(data[i,1]),int(data[i,2])))

f = open('nicf_base.txt','w+')
for key, value in uids_consumes.items():
    f.write(f'{key}\t'+'\t'.join([f'{iid}:{r}'for iid,r in value]))
    f.write('\n')
f.close()

# iids = dict()
# for i, iid in enumerate(np.unique(data[:, 1])):
    # iids[iid] = i

# data[:, 1] = np.vectorize(lambda x: iids[x])(data[:, 1])
# data[:, 0] = data[:, 0] - 1
