import inquirer
import interactors
import numpy as np
import os
import mf
import evaluation_policy
from utils.PersistentDataManager import PersistentDataManager
from .InteractorCache import InteractorCache
import utils.util as util
import ctypes
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time


class InteractorRunner():

    def __init__(self, dm, interactors_general_settings,
                 interactors_preprocessor_paramaters,
                 evaluation_policies_parameters):
        self.dm = dm
        self.interactors_general_settings = interactors_general_settings
        self.interactors_preprocessor_paramaters = interactors_preprocessor_paramaters
        self.evaluation_policies_parameters = evaluation_policies_parameters

    def get_interactor_name(self, interactor_class_name):
        return self.interactors_general_settings[interactor_class_name]['name']

    def select_interactors(self):
        pdm = PersistentDataManager(directory='state_save')
        choices = [
            v['name'] for v in self.interactors_general_settings.values()
        ]
        if pdm.file_exists('interactors_selection_cache'):
            interactors_selection_cache = pdm.load(
                'interactors_selection_cache')
            for i in reversed(interactors_selection_cache):
                choices.remove(i)
                choices.insert(0, i)
        else:
            print("No cache in interactors selection")
        q = [
            inquirer.Checkbox('interactors',
                              message='Interactors to run',
                              choices=choices)
        ]
        answers = inquirer.prompt(q)

        if pdm.file_exists('interactors_selection_cache'):
            pdm.save(
                'interactors_selection_cache',
                list(
                    OrderedDict.fromkeys(answers['interactors'] +
                                         interactors_selection_cache)))
        else:
            pdm.save('interactors_selection_cache', answers['interactors'])

        interactors_class_names = dict(
            zip([v['name'] for v in self.interactors_general_settings.values()],
                self.interactors_general_settings.keys()))

        interactors_classes = list(
            map(lambda x: eval('interactors.' + interactors_class_names[x]),
                answers['interactors']))
        return interactors_classes

    def create_interactor(self, itr_class):
        if self.interactors_preprocessor_paramaters[
                self.dm.dataset_preprocessor.name][
                    itr_class.__name__] != None and 'parameters' in self.interactors_preprocessor_paramaters[
                        self.dm.dataset_preprocessor.name][itr_class.__name__]:
            parameters = self.interactors_preprocessor_paramaters[
                self.dm.dataset_preprocessor.name][
                    itr_class.__name__]['parameters']
        else:
            parameters = {}
        #     parameters =

        # print(self.interactors_preprocessor_paramaters[self.dm.dataset_preprocessor.name][itr_class.__name__])
        itr = itr_class(**parameters)
        return itr

    def get_interactors_evaluation_policy(self):
        evaluation_policy_name = open(
            "settings/interactors_evaluation_policy.txt").read().replace(
                '\n', '')
        evaluation_policy = eval('evaluation_policy.' + evaluation_policy_name)(
            **self.evaluation_policies_parameters[evaluation_policy_name])
        return evaluation_policy

    def run_interactor(self, itr, forced_run):
        # print("11111")
        pdm = PersistentDataManager(directory='results')
        evaluation_policy = self.get_interactors_evaluation_policy()
        if forced_run or not pdm.file_exists(InteractorCache().get_id(
                self.dm, evaluation_policy, itr)):
            # print("22222")
            # print(self.dm)
            # print(self.dm.dataset_preprocessed[0])
            # print(self.dm.dataset_preprocessed[1])
            history_items_recommended = evaluation_policy.evaluate(
                itr, self.dm.dataset_preprocessed[0],
                self.dm.dataset_preprocessed[1])
            # print("33333")

            pdm = PersistentDataManager(directory='results')
            pdm.save(InteractorCache().get_id(self.dm, evaluation_policy, itr),
                     history_items_recommended)
        else:
            print("Already executed",
                  InteractorCache().get_id(self.dm, evaluation_policy, itr))

    @staticmethod
    def _run_interactor(obj_id, itr, forced_run):
        self = ctypes.cast(obj_id, ctypes.py_object).value
        self.run_interactor(itr, forced_run)

    def run_interactors(self, interactors_classes, forced_run=False,parallel=False):

        args = [(id(self), self.create_interactor(itr_class), forced_run)
                for itr_class in interactors_classes]
        if parallel:
            util.run_parallel(self._run_interactor, args)
        else:
            for i, itr_class in enumerate(interactors_classes):
                self._run_interactor(*args[i])

    def run_interactors_search(self,
                               interactors_classes,
                               interactors_search_parameters,
                               num_tasks=None,
                               forced_run=False):
        # print("ewq ejiwqeijqw iewq jieqw jewqjieqwi")
        if num_tasks == None:
            num_tasks = os.cpu_count()
            # print("ewqewjiqewjiewijq ijewqijwqe",num_tasks)

        with ProcessPoolExecutor() as executor:
            futures = set()
            for itr_class in interactors_classes:
                for parameters in interactors_search_parameters[
                        itr_class.__name__]:
                    f = executor.submit(self._run_interactor, id(self),
                                        itr_class(**parameters), forced_run)
                    futures.add(f)

                    if len(futures) >= num_tasks:
                        completed, futures = wait(futures,
                                                  return_when=FIRST_COMPLETED)

            for f in futures:
                f.result()
