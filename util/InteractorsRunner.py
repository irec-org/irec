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
            pmf_model.load_var(dsf.consumption_matrix[dsf.train_uids])
            pmf_model = pmf_model.load()

        if np.any(list(map(
                lambda itr_class: itr_class in
                    [interactors.LinUCB,
                    interactors.MostRepresentative,
                    interactors.LinEGreedy,
                    interactors.UCBLearner,
                    interactors.OurMethod1],
                interactors_classes
                ))):
            print('Loading SVD')
            svd_model = mf.SVD(name_prefix=dsf.base)
            svd_model = svd_model.load()
            Q = svd_model.items_weights

        for itr_class in interactors_classes:
            if issubclass(itr_class,interactors.ICF):
                itr = itr_class(var=pmf_model.var,
                                user_lambda=pmf_model.user_lambda,
                                consumption_matrix=dsf.consumption_matrix,
                                name_prefix=dsf.base
                )
            else:
                itr = itr_class(consumption_matrix=dsf.consumption_matrix,
                                name_prefix=dsf.base
                )

            if itr_class in [interactors.LinearThompsonSampling]:
                itr.interact(dsf.test_uids, pmf_model.items_means, pmf_model.items_covs)
            elif issubclass(itr_class,interactors.ICF):
                itr.interact(dsf.test_uids, pmf_model.items_means)
            elif itr_class in [interactors.LinUCB,
                            interactors.LinEGreedy,
                            interactors.UCBLearner,
                            interactors.MostRepresentative,
                            interactors.OurMethod1]:
                itr.interact(dsf.test_uids,Q)
            else:
                itr.interact(dsf.test_uids)
