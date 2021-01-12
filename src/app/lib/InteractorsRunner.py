import inquirer
import interactors
import numpy as np
import mf
import evaluation_policy

class InteractorsRunner():

    def __init__(self,dm,interactors_general_settings,interactors_preprocessor_paramaters,evaluation_policies_parameters):
        self.dm = dm
        self.interactors_general_settings = interactors_general_settings
        self.interactors_preprocessor_paramaters = interactors_preprocessor_paramaters
        self.evaluation_policies_parameters = evaluation_policies_parameters

    def select_interactors(self):
        q = [
            inquirer.Checkbox('interactors',
                              message='Interactors to run',
                              choices=[v['name'] for v in self.interactors_general_settings.values()]
            )
        ]
        answers=inquirer.prompt(q)
        interactors_class_names = dict(zip([v['name'] for v in self.interactors_general_settings.values()],self.interactors_general_settings.keys()))
        # interactors_names = dict(zip(self.interactors_general_settings.keys(),[v['name'] for v in self.interactors_general_settings.values()]))

        interactors_classes = list(map(lambda x:eval('interactors.'+interactors_class_names[x]),answers['interactors']))
        self.interactors_classes = interactors_classes
        return interactors_classes

    def create_and_run_interactor(self,itr_class):
        # print(self.interactors_preprocessor_paramaters[self.dm.dataset_preprocessor.name])
        # print(itr_class.__name__)
        
        if self.interactors_preprocessor_paramaters[self.dm.dataset_preprocessor.name][itr_class.__name__] != None and 'parameters' in self.interactors_preprocessor_paramaters[self.dm.dataset_preprocessor.name][itr_class.__name__]:
            parameters = self.interactors_preprocessor_paramaters[self.dm.dataset_preprocessor.name][itr_class.__name__]['parameters']
        else:
            parameters = {}
        #     parameters = 
        
        # print(self.interactors_preprocessor_paramaters[self.dm.dataset_preprocessor.name][itr_class.__name__])
        itr = itr_class(**parameters)
        itr_evaluation_policy=self.interactors_general_settings[itr_class.__name__]['evaluation_policy']
        evaluation_policy = eval('evaluation_policy.'+itr_evaluation_policy)(**self.evaluation_policies_parameters[itr_evaluation_policy])
        evaluation_policy.evaluate(itr,self.dm.dataset_preprocessed[0],self.dm.dataset_preprocessed[1])
        pass

    def run_interactors(self):
        for itr_class in self.interactors_classes:
            self.create_and_run_interactor(itr_class)
