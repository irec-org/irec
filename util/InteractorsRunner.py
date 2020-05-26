import inquirer
import interactors
import numpy as np
import mf

class InteractorsRunner():

    def __init__(self,dsf,interactors_classes=None):
        self.dsf = dsf
        self.interactors_classes = interactors_classes

    def select_interactors(self):
        q = [
            inquirer.Checkbox('interactors',
                              message='Interactors to run',
                              choices=list(interactors.INTERACTORS.keys())
            )
        ]
        answers=inquirer.prompt(q)
        interactors_classes = list(map(lambda x:interactors.INTERACTORS[x],answers['interactors']))
        self.interactors_classes = interactors_classes
        return interactors_classes

    def create_and_run_interactor(self,itr_class):
        dsf = self.dsf

        if issubclass(itr_class,interactors.ICF):
            itr = itr_class(var=self.pmf_model.var,
                            user_lambda=self.pmf_model.get_user_lambda(),
                            test_consumption_matrix=dsf.test_consumption_matrix,
                            train_consumption_matrix=dsf.train_consumption_matrix,
                            name_prefix=dsf.base
            )
        else:
            itr = itr_class(name_prefix=dsf.base,
                            test_consumption_matrix=dsf.test_consumption_matrix,
                            train_consumption_matrix=dsf.train_consumption_matrix,
            )

        if itr_class in [interactors.LinearThompsonSampling]:
            itr.interact(self.pmf_model.items_means, self.pmf_model.items_covs)
        elif issubclass(itr_class,interactors.ICF):
            itr.interact(self.pmf_model.items_means)
        elif itr_class in [interactors.LinUCB,
                        interactors.LinEGreedy,
                        interactors.UCBLearner,
                        interactors.MostRepresentative,
                        interactors.OurMethod1]:
            itr.interact(self.mf_model.items_weights)
        else:
            itr.interact()

    def run_interactors(self):
        self.select_interactors()
        dsf = self.dsf
        interactors_classes = self.interactors_classes
        is_spmatrix = dsf.is_spmatrix

        if np.any(list(map(
                lambda itr_class: issubclass(itr_class,interactors.ICF),
                interactors_classes))):
            if not is_spmatrix:
                pmf_model = mf.ICFPMF(name_prefix=dsf.base)
            else:
                pmf_model = mf.ICFPMFS(name_prefix=dsf.base)
            print('Loading %s'%(pmf_model.__class__.__name__))
            # pmf_model.load_var(dsf.train_consumption_matrix)
            pmf_model = pmf_model.load()
            self.pmf_model = pmf_model

        if np.any(list(map(
                lambda itr_class: itr_class in
                    [interactors.LinUCB,
                    interactors.MostRepresentative,
                    interactors.LinEGreedy,
                    interactors.UCBLearner,
                    interactors.OurMethod1],
                interactors_classes
                ))):
            # print('Loading PMF')
            # mf_model = mf.PMF(name_prefix=dsf.base)
            # mf_model = mf_model.load()
            # mf_model.items_weights
            print('Loading SVD')
            mf_model = mf.SVD(name_prefix=dsf.base)
            mf_model = mf_model.load()
            self.mf_model = mf_model

        for itr_class in interactors_classes:
            self.create_and_run_interactor(itr_class)

    
