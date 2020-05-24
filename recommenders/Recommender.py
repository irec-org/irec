from util import Saveable
class Recommender(Saveable):
    def __init__(self,training_matrix,result_list_size=10,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.training_matrix = training_matrix
        self.result_list_size = result_list_size

    def train(self):
        print('Training '+self.get_verbose_name())

    def predict(self):
        print('Predicting '+self.get_verbose_name())
        pass
