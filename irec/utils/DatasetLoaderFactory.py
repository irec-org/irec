from . import DatasetLoader
import irec.utils.dataset as dataset


class DatasetLoaderFactory:
    def create(self, dataset_name, dataset_parameters):
        dl = None
        if dataset_name == "MovieLens 100k O":
            dl = DatasetLoader.ML100kDatasetLoader(**dataset_parameters)
        elif dataset_name in [
            "MovieLens 1M",
            "MovieLens 10M",
            "MovieLens 20M",
            "LastFM 5k",
            "Kindle Store",
            "Kindle Store 4k",
            "Netflix 10k",
            "Good Books",
            "Yahoo Music",
            "Good Reads 10k",
        ]:
            dl = DatasetLoader.DefaultDatasetLoader(dataset.DefaultDataset(),**dataset_parameters)
        elif dataset_name == "Netflix":
            dl = DatasetLoader.DefaultDatasetLoader(dataset.Netflix(),**dataset_parameters)
        elif dataset_name in [
            "MovieLens 1M Validation",
            "MovieLens 10M Validation",
            "MovieLens 20M Validation",
            "LastFM 5k Validation",
            "Kindle Store Validation",
            "Kindle Store 4k Validation",
            "Good Books Validation",
            "Yahoo Music Validation",
            "Good Reads 10k Validation",

        ]:
            dl = DatasetLoader.DefaultValidationDatasetLoader(dataset.DefaultDataset(),**dataset_parameters)

        elif dataset_name in ["Netflix Validation","Netflix 10k Validation"]:
            dl = DatasetLoader.DefaultDatasetLoader(dataset.Netflix(),**dataset_parameters)
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
