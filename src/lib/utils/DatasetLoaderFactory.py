from . import DatasetLoader

class DatasetLoaderFactory:
    def create(self,dataset_name,dataset_parameters):
        dl = None
        if dataset_name == 'MovieLens 100k O':
            dl=DatasetLoader.ML100kDatasetLoader(**dataset_parameters)
        else:
            raise IndexError
        return dl
