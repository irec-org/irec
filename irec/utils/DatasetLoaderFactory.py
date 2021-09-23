from . import DatasetLoader

class DatasetLoaderFactory:
	def create(self,dataset_name,dataset_parameters):
		dl = None
		if dataset_name == 'MovieLens 100k O':
			dl=DatasetLoader.ML100kDatasetLoader(**dataset_parameters)
		elif dataset_name == 'LastFM 5k':
			dl=DatasetLoader.LastFM5kDatasetLoader(**dataset_parameters)
		elif dataset_name == 'LastFM 5k Validation':
			dl=DatasetLoader.LastFM5kValidationDatasetLoader(**dataset_parameters)
		else:
			raise IndexError
		return dl
