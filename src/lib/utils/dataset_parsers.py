import os
# from dataset import *
from . import dataset
from copy import copy

class DatasetParser:
    pass

class TRTE(DatasetParser):
    def parse_dataset(self,dataset_descriptor):
        base_dir = dataset_descriptor.base_dir
        train_data = np.loadtxt(os.path.join(base_dir,'train.data'),delimiter='::')
        test_data = np.loadtxt(os.path.join(base_dir,'test.data'),delimiter='::')

        dataset = Dataset(np.vstack([train_data,test_data]))
        dataset.update_from_data()
        train_dataset = copy(dataset)
        train_dataset.data = train_data
        test_dataset = copy(dataset)
        test_dataset.data = test_data
        return train_dataset, test_dataset

class MovieLens100k(DatasetParser):
    def parse_dataset(self,dataset_descriptor):
        base_dir = dataset_descriptor.base_dir
        data = np.loadtxt(os.path.join(base_dir,'u.data'),delimiter='\t')
        data[0] = data[0] - 1
        data[1] = data[1] - 1
        dataset = Dataset(data)
        dataset.update_from_data()
        return dataset

class MovieLens1M(DatasetParser):
    def parse_dataset(self,dataset_descriptor):
        base_dir = dataset_descriptor.base_dir
        data = np.loadtxt(os.path.join(base_dir,'ratings.dat'),delimiter='::')
        iids = dict()
        for i, iid in enumerate(df_cons[1].unique()):
            iids[iid] = i
        data[1] = np.vectorize(lambda x: iids[x])(data[1])
        data[0] = data[0] - 1
        dataset = Dataset(data)
        dataset.update_from_data()
        return dataset
