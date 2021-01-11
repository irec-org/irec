from os.path import dirname, realpath, sep, pardir
import os
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + pardir + sep + "lib")

import inquirer
import utils.dataset as dataset
import yaml
# import utils.dataset_parsers as dataset_parsers
from lib.DirectoryDependent import DirectoryDependent
import utils.splitters as splitters
import utils.util as util
import pickle

import inquirer
class DatasetManager:

    def __init__(self):
        pass

    def get_datasets_preprocessors_settings(self):
        with open("settings"+sep+"datasets_preprocessors.yaml") as f:
            self.loader = yaml.SafeLoader
            datasets_preprocessors = yaml.load(f,Loader=self.loader)

            datasets_preprocessors = {setting['name']:setting
                                      for setting in datasets_preprocessors}
            return datasets_preprocessors

    def request_dataset_preprocessor(self):

        datasets_preprocessors = self.get_datasets_preprocessors_settings()
        q = [
            inquirer.List(0,
                          message='Datasets Preprocessors',
                          choices=datasets_preprocessors.keys()
                          )
        ]
        answers=inquirer.prompt(q)
        self.dataset_preprocessor = datasets_preprocessors[answers[0]]
    def initialize_engines(self):
        self.dataset_parser = eval('dataset.'+self.dataset_preprocessor['dataset_parser'])()
        if self.dataset_preprocessor['splitter'] != None:
            with open("settings"+sep+"splitters.yaml") as splittersf:
                self.splitters_settings = yaml.load(splittersf,Loader=self.loader)
                self.splitter = eval('splitters.'+self.dataset_preprocessor['splitter'])(**self.splitters_settings[self.dataset_preprocessor['splitter']])

    def run_parser(self):
        self.dataset_descriptor=dataset.DatasetDescriptor(
            self.dataset_preprocessor['name'],
            os.path.join(
                DirectoryDependent().DIRS['datasets'],
                self.dataset_preprocessor['dataset_dir']))
        
        self.dataset_parsed = self.dataset_parser.parse_dataset(self.dataset_descriptor)
    def run_splitter(self):
        if self.dataset_preprocessor['splitter'] != None:
            # with open("settings"+sep+"splitters.yaml") as splittersf:
            #     self.splitters_settings = yaml.load(splittersf,Loader=self.loader)
            #     self.splitter = eval('splitters.'+self.dataset_preprocessor['splitter'])(**self.splitters_settings[self.dataset_preprocessor['splitter']])
            self.result=self.splitter.apply(self.dataset_parsed)
        else:
            self.result = dataset_parsed

    def save(self):
        # print(open(self.get_file_name(),'wb'))
        pickle.dump(self.result,open(self.get_file_name(),'wb'))
        pass

    def load(self):
        self.result = pickle.load(open(self.get_file_name(),'rb'))
        return self.result

    def get_file_name(self):
        # print(os.path.join(self.get_id()+'.pickle'))
        return os.path.join(DirectoryDependent().DIRS['dataset_preprocess'],os.path.join(self.get_id()+'.pickle'))
    def get_id(self):
        return 'dspp_'+\
            self.dataset_descriptor.get_id()+','+\
            self.dataset_parser.get_id()+\
            (','+self.splitter.get_id()) if self.dataset_preprocessor['splitter'] != None else ''
