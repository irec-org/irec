from . import dataset
import random
import numpy as np


class DatasetLoader:
    pass


class ML100kDatasetLoader:
    def __init__(
        self, dataset_path, crono, random_seed, test_consumes, train_size
    ) -> None:
        self.dataset_path = dataset_path
        self.crono = crono
        self.random_seed = random_seed
        self.test_consumes = test_consumes
        self.train_size = train_size

    def load(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        ml100k_processor = dataset.MovieLens100k()
        data = ml100k_processor.process(self.dataset_path)

        traintest_processor = dataset.TrainTestConsumption(
            crono=self.crono,
            test_consumes=self.test_consumes,
            train_size=self.train_size,
        )
        res = traintest_processor.process(data)
        return res


class LastFM5kDatasetLoader:
    def __init__(
        self, dataset_path, crono, random_seed, test_consumes, train_size
    ) -> None:
        self.dataset_path = dataset_path
        self.crono = crono
        self.random_seed = random_seed
        self.test_consumes = test_consumes
        self.train_size = train_size

    def load(self):
        np.random.seed(self.random_seed)
        lastfm5k_processor = dataset.LastFM5k()
        data = lastfm5k_processor.process(self.dataset_path)

        traintest_processor = dataset.TrainTestConsumption(
            crono=self.crono,
            test_consumes=self.test_consumes,
            train_size=self.train_size,
        )
        res = traintest_processor.process(data)
        return res


class LastFM5kValidationDatasetLoader:
    def __init__(
        self, dataset_path, crono, random_seed, test_consumes, train_size
    ) -> None:
        self.dataset_path = dataset_path
        self.crono = crono
        self.random_seed = random_seed
        self.test_consumes = test_consumes
        self.train_size = train_size

    def load(self):
        np.random.seed(self.random_seed)
        lastfm5k_processor = dataset.LastFM5k()
        data = lastfm5k_processor.process(self.dataset_path)

        traintest_processor = dataset.TrainTestConsumption(
            crono=self.crono,
            test_consumes=self.test_consumes,
            train_size=self.train_size,
        )
        res = traintest_processor.process(traintest_processor.process(data).train)

        return res


class TRTEDatasetLoader:
    def __init__(
        self,
        dataset_path
        # self, dataset_path, crono, random_seed, test_consumes, train_size
    ) -> None:
        self.dataset_path = dataset_path
        # self.crono = crono
        # self.random_seed = random_seed
        # self.test_consumes = test_consumes
        # self.train_size = train_size

    def load(self):
        # np.random.seed(self.random_seed)
        trte_processor = dataset.TRTE()
        data = trte_processor.process(self.dataset_path)
        res = data
        # traintest_processor = dataset.TrainTestConsumption(
        # crono=self.crono,
        # test_consumes=self.test_consumes,
        # train_size=self.train_size,
        # )
        # res = traintest_processor.process(data)
        return res


class TRTEValidationDatasetLoader:
    def __init__(
        self, dataset_path, crono, random_seed, test_consumes, train_size
    ) -> None:
        self.dataset_path = dataset_path
        self.crono = crono
        self.random_seed = random_seed
        self.test_consumes = test_consumes
        self.train_size = train_size

    def load(self):
        np.random.seed(self.random_seed)
        trte_processor = dataset.TRTE()
        data = trte_processor.process(self.dataset_path)

        traintest_processor = dataset.TrainTestConsumption(
            crono=self.crono,
            test_consumes=self.test_consumes,
            train_size=self.train_size,
        )
        res = traintest_processor.process(data.train)

        return res