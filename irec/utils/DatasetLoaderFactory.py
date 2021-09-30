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
        elif dataset_name in [
            "Kindle 4k TRTE",
            "Good Books TRTE",
            "LastFM 5k TRTE",
            "MovieLens 10M TRTE",
            "Yahoo Music TRTE",
            "GoodBooks TRTE",
            "LastFM 5k TRTE",
            "Netflix 10k TRTE",
        ]:
            dl = DatasetLoader.TRTEDatasetLoader(**dataset_parameters)
        elif dataset_name in [
            "Kindle 4k TRTE Validation",
            "Good Books TRTE Validation",
            "LastFM 5k TRTE Validation",
            "GoodBooks TRTE Validation",
            "Yahoo Music TRTE Validation",
            "LastFM 5k TRTE Validation",
            "MovieLens 10M TRTE Validation",
            "Netflix 10k TRTE Validation",
        ]:
            dl = DatasetLoader.TRTEValidationDatasetLoader(**dataset_parameters)
        else:
            raise IndexError(dataset_name)
        return dl
