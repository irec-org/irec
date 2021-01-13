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

import inquirer
class DatasetManager:

    def __init__(self):
        pass

    def get_datasets_preprocessors_settings(self):
        with open("settings"+sep+"datasets_preprocessors_parameters.yaml") as f:
            self.loader = yaml.SafeLoader
            datasets_preprocessors = yaml.load(f,Loader=self.loader)

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
        self.dataset_preprocessor = datasets_preprocessors[answers['dspp']]
        
    def initialize_engines(self):
        dataset_parser = eval('dataset.'+self.dataset_preprocessor['preprocessor']['dataset_parser'])()
        if self.dataset_preprocessor['preprocessor']['splitter'] != None:
            with open("settings"+sep+"splitters_parameters.yaml") as splittersf:
                splitters_settings = yaml.load(splittersf,Loader=self.loader)
                splitter = eval('splitters.'+self.dataset_preprocessor['preprocessor']['splitter'])(**splitters_settings[self.dataset_preprocessor['preprocessor']['splitter']])
        else:
            splitter = None

        dataset_descriptor=dataset.DatasetDescriptor(
            os.path.join(
                DirectoryDependent().DIRS['datasets'],
                self.dataset_preprocessor['dataset_descriptor']['dataset_dir']))
        preprocessor = dataset.Preprocessor(
            dataset_parser,splitter
        )
        self.dataset_preprocessor = dataset.DatasetPreprocessor(self.dataset_preprocessor['name'],dataset_descriptor,preprocessor)


    def run_parser(self):
        self.dataset_parsed = self.dataset_preprocessor.preprocessor.dataset_parser.parse_dataset(self.dataset_preprocessor.dataset_descriptor)
    def run_splitter(self):
        if self.dataset_preprocessor.preprocessor.splitter != None:
            # with open("settings"+sep+"splitters.yaml") as splittersf:
            #     self.splitters_settings = yaml.load(splittersf,Loader=self.loader)
            #     self.splitter = eval('splitters.'+self.dataset_preprocessor['splitter'])(**self.splitters_settings[self.dataset_preprocessor['splitter']])
            self.dataset_preprocessed=self.dataset_preprocessor.preprocessor.splitter.apply(self.dataset_parsed)
        else:
            self.dataset_preprocessed = dataset_parsed

        # self.train_dataset = self.dataset_preprocessed[0]
        # self.test_dataset = self.dataset_preprocessed[1]

    def save(self):
        print(self.get_file_name())
        # print(open(self.get_file_name(),'wb'))
        # Path('/'.join(self.get_file_name().split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        pdm = PersistentDataManager('dataset_preprocess')
        pdm.save(self.get_file_name(),self.dataset_preprocessed)
        # util.create_path_to_file(self.get_file_name())
        # pickle.dump(self.dataset_preprocessed,open(,'wb'))
        # pass

    def load(self):

        pdm = PersistentDataManager('dataset_preprocess')
        self.dataset_preprocessed = pdm.load(self.get_file_name())
        return self.dataset_preprocessed
        # self.dataset_preprocessed = pickle.load(open(self.get_file_name(),'rb'))
        # return self.dataset_preprocessed

    def get_file_name(self):
        # print(os.path.join(self.get_id()+'.pickle'))
        return os.path.join(DirectoryDependent().DIRS['dataset_preprocess'],os.path.join(self.get_id()+'.pickle'))
    def get_id(self):
        return 'dspp_'+self.dataset_preprocessor.get_id()
            # self.dataset_parser.get_id()+\
            # (','+self.splitter.get_id()) if self.dataset_preprocessor['splitter'] != None else ''
