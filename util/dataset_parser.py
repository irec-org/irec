import os

class DatasetParser:
    pass

class TRTE(DatasetParser):
    def parse_dataset(self,dataset_descriptor):
        base_dir = dataset_descriptor.base_dir
        train_data = np.loadtxt(os.path.join(base_dir,'train.data'),delimiter='::')
        test_data = np.loadtxt(os.path.join(base_dir,'test.data'),delimiter='::')
        return train_data, test_data


class MovieLens100k(DatasetParser):
    def parse_dataset(self,dataset_descriptor):
        base_dir = dataset_descriptor.base_dir
        data = np.loadtxt(os.path.join(base_dir,'u.data'),delimiter='\t')
        data[0] = data[0] - 1
        data[1] = data[1] - 1
        return data

class MovieLens1M(DatasetParser):
    def parse_dataset(self,dataset_descriptor):
        base_dir = dataset_descriptor.base_dir
        data = np.loadtxt(os.path.join(base_dir,'ratings.dat'),delimiter='::')
        iids = dict()
        for i, iid in enumerate(df_cons[1].unique()):
            iids[iid] = i
        data[1] = np.vectorize(lambda x: iids[x])(data[1])
        data[0] = data[0] - 1
        return data
