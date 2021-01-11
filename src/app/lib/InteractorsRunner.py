import inquirer
import interactors
import numpy as np
import mf

class InteractorsRunner():

    def __init__(self,dm,interactors_names,interactors_preprocessor_paramaters):
        self.dm = dm
        self.interactors_names = interactors_names
        self.interactors_preprocessor_paramaters = interactors_preprocessor_paramaters

    def select_interactors(self):
        
        q = [
            inquirer.Checkbox('interactors',
                              message='Interactors to run',
                              choices=list(self.interactors_names.values())
            )
        ]
        answers=inquirer.prompt(q)
        # ddd = dict()
        # for interactor_setting in self.interactors_names:
        #     # list(interactor_setting.keys())[0]
        #     ddd[]
        interactors_class_names = dict(zip(self.interactors_names.values(),self.interactors_names.keys()))

            
        interactors_classes = list(map(lambda x:eval('interactors.'+interactors_class_names[x]),answers['interactors']))
        self.interactors_classes = interactors_classes
        return interactors_classes

    # def run_bases(self,bases):
    #     for base in bases:
    #         dm = DatasetFormatter(base=base)
    #         dm = dm.load()
    #         self.dm = dm
    #         self.run_interactors()

    def create_and_run_interactor(self,itr_class):
        dm = self.dm

        if issubclass(itr_class,interactors.ICF):
            itr = itr_class(var=self.pmf_model.var,
                            user_lambda=self.pmf_model.get_user_lambda(),
                            test_consumption_matrix=dm.test_consumption_matrix,
                            train_consumption_matrix=dm.train_consumption_matrix,
                            name_prefix=dm.base
            )
        else:
            itr = itr_class(name_prefix=dm.base,
                            test_consumption_matrix=dm.test_consumption_matrix,
                            train_consumption_matrix=dm.train_consumption_matrix,
            )

        if itr_class in [interactors.LinearThompsonSampling]:
            itr.interact(self.pmf_model.items_means, self.pmf_model.items_covs)
        elif issubclass(itr_class,interactors.ICF):
            itr.interact(self.pmf_model.items_means)
        elif itr_class in [interactors.LinUCB,
                        interactors.LinEGreedy,
                        interactors.UCBLearner,
                        interactors.MostRepresentative,
                        interactors.OurMethod1,
                        interactors.OurMethod2,
                        interactors.COFIBA]:
            itr.interact(self.mf_model.items_weights)
        else:
            itr.interact()

    def run_interactors(self):
        dm = self.dm
        interactors_classes = self.interactors_classes
        is_spmatrix = dm.is_spmatrix

        if np.any(list(map(
                lambda itr_class: issubclass(itr_class,interactors.ICF),
                interactors_classes))):
            if not is_spmatrix:
                pmf_model = mf.ICFPMF(name_prefix=dm.base)
            else:
                pmf_model = mf.ICFPMFS(name_prefix=dm.base)
            print('Loading %s'%(pmf_model.__class__.__name__))
            pmf_model.load_var(dm.train_consumption_matrix)
            pmf_model = pmf_model.load()
            self.pmf_model = pmf_model

        if np.any(list(map(
                lambda itr_class: itr_class in
                    [interactors.LinUCB,
                    interactors.MostRepresentative,
                    interactors.LinEGreedy,
                    interactors.UCBLearner,
                    interactors.OurMethod1,
                    interactors.OurMethod2,
                     interactors.COFIBA],
                interactors_classes
                ))):
            # if not is_spmatrix:
            #     mf_model = mf.ICFPMF(name_prefix=dm.base)
            # else:
            #     mf_model = mf.ICFPMFS(name_prefix=dm.base)
            # print('Loading %s'%(mf_model.__class__.__name__))
            # mf_model.load_var(dm.train_consumption_matrix)
            # mf_model = mf_model.load()
            # self.mf_model = mf_model
            # print('Loading PMF')
            # mf_model = mf.PMF(name_prefix=dm.base)
            # mf_model.load_var(dm.train_consumption_matrix)
            # mf_model = mf_model.load()
            print('Loading SVD')
            mf_model = mf.SVD(name_prefix=dm.base)
            mf_model = mf_model.load()
            self.mf_model = mf_model

        for itr_class in interactors_classes:
            self.create_and_run_interactor(itr_class)

    
