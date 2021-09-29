from . import DatasetLoader


class DatasetLoaderFactory:
    def create(self, dataset_name, dataset_parameters):
        dl = None
        if dataset_name == "MovieLens 100k O":
            dl = DatasetLoader.ML100kDatasetLoader(**dataset_parameters)
        elif dataset_name == "LastFM 5k":
            dl = DatasetLoader.LastFM5kDatasetLoader(**dataset_parameters)
        elif dataset_name == "LastFM 5k Validation":
            dl = DatasetLoader.LastFM5kValidationDatasetLoader(**dataset_parameters)
        elif dataset_name == "LastFM 5k TRTE":
            dl = DatasetLoader.TRTEDatasetLoader(**dataset_parameters)
        elif dataset_name == "LastFM 5k TRTE Validation":
            dl = DatasetLoader.TRTEValidationDatasetLoader(**dataset_parameters)
        elif dataset_name == "MovieLens 10M TRTE":
            dl = DatasetLoader.TRTEDatasetLoader(**dataset_parameters)
        elif dataset_name == "MovieLens 10M TRTE Validation":
            dl = DatasetLoader.TRTEValidationDatasetLoader(**dataset_parameters)
        elif dataset_name == "Yahoo Music TRTE":
            dl = DatasetLoader.TRTEDatasetLoader(**dataset_parameters)
        elif dataset_name == "Yahoo Music TRTE Validation":
            dl = DatasetLoader.TRTEValidationDatasetLoader(**dataset_parameters)
        elif dataset_name == "GoodBooks TRTE":
            dl = DatasetLoader.TRTEDatasetLoader(**dataset_parameters)
        elif dataset_name == "GoodBooks TRTE Validation":
            dl = DatasetLoader.TRTEValidationDatasetLoader(**dataset_parameters)
        elif dataset_name == "LastFM 5k TRTE":
            dl = DatasetLoader.TRTEDatasetLoader(**dataset_parameters)
        elif dataset_name == "LastFM 5k TRTE Validation":
            dl = DatasetLoader.TRTEValidationDatasetLoader(**dataset_parameters)
        elif dataset_name == "Kindle 4k TRTE":
            dl = DatasetLoader.TRTEDatasetLoader(**dataset_parameters)
        elif dataset_name == "Kindle 4k TRTE Validation":
            dl = DatasetLoader.TRTEValidationDatasetLoader(**dataset_parameters)
        else:
            raise IndexError
        return dl
