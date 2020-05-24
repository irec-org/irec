from util import Saveable
from collections import defaultdict
class Recommender(Saveable):
    def __init__(self,result_list_size=10,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.result_list_size = result_list_size
        self.results = defaultdict(list)

    def train(self):
        print('Training '+self.get_verbose_name())

    def predict(self):
        print('Predicting '+self.get_verbose_name())
        self.results.clear()
