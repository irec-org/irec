from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + pardir + sep + "lib")

from pathlib import Path
import inquirer
import utils.dataset as dataset
import yaml
# import utils.dataset_parsers as dataset_parsers
from .DirectoryDependent import DirectoryDependent
import utils.splitters as splitters
import utils.util as util
import pickle
from .PersistentDataManager import PersistentDataManager
from .Parameterizable import Parameterizable

import inquirer
class DatasetManager(Parameterizable):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.parameters.extend(['dataset_preprocessor'])

    def get_datasets_preprocessors_settings(self):
        with open("settings"+sep+"datasets_preprocessors_parameters.yaml") as f:
            loader = yaml.SafeLoader
            datasets_preprocessors = yaml.load(f,Loader=loader)

            datasets_preprocessors = {setting['name']: setting
                                      for setting in datasets_preprocessors}
            return datasets_preprocessors

    def request_dataset_preprocessor(self):

        datasets_preprocessors = self.get_datasets_preprocessors_settings()
        q = [
            inquirer.List('dspp',
                          message='Datasets Preprocessors',
                          choices=list(datasets_preprocessors.keys())
                          )
        ]
        answers=inquirer.prompt(q)
        return datasets_preprocessors[answers['dspp']]

    def request_datasets_preprocessors(self):

        datasets_preprocessors = self.get_datasets_preprocessors_settings()
        q = [
            inquirer.Checkbox('dspp',
                          message='Datasets Preprocessors',
                          choices=list(datasets_preprocessors.keys())
                          )
        ]
        answers=inquirer.prompt(q)
        return list(map(lambda x: datasets_preprocessors[x], answers['dspp']))
        
    def initialize_engines(self,dataset_preprocessor):
        pipeline = dataset.Pipeline()
        with open("settings"+sep+"processors_parameters.yaml") as processors_parameters_f:
            loader = yaml.SafeLoader
            processors_parameters = yaml.load(processors_parameters_f,Loader=loader)
            for data_processor in dataset_preprocessor['preprocessor']:
                if data_processor in processors_parameters:
                    pipeline.steps.append(eval('dataset.'+data_processor)(**processors_parameters[data_processor]))
                else:
                    pipeline.steps.append(eval('dataset.'+data_processor)())
        dataset_descriptor=dataset.DatasetDescriptor(
            os.path.join(
                DirectoryDependent().DIRS['datasets'],
                dataset_preprocessor['dataset_descriptor']['dataset_dir']))
        preprocessor = pipeline
        self.dataset_preprocessor = dataset.DatasetPreprocessor(dataset_preprocessor['name'],dataset_descriptor,preprocessor)

    def run_preprocessor(self):
        self.dataset_preprocessed = self.dataset_preprocessor.preprocessor.process(self.dataset_preprocessor.dataset_descriptor)
        self.train_dataset = self.dataset_preprocessed[0]
        self.test_dataset = self.dataset_preprocessed[1]

    def save(self):
        pdm = PersistentDataManager('dataset_preprocess')
        pdm.save(self.get_id(),self.dataset_preprocessed)

    def load(self):
        pdm = PersistentDataManager('dataset_preprocess')
        self.dataset_preprocessed = pdm.load(self.get_id())
        self.train_dataset = self.dataset_preprocessed[0]
        self.test_dataset = self.dataset_preprocessed[1]
        return self.dataset_preprocessed
