from . import Recommender
import mf
class SVDPlusPlus(Recommender):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        pass

    def train(self,train_matrix):
        mf_model = mf.SVDPlusPlus()
        mf_model.fit(train_matrix)
        self.b_u, self.b_i, self.p, self.q, self.y =\
            mf_model.b_u,mf_model.b_i,\
            mf_model.p,mf_model.q,\
            mf_model.y

    def predict(self,test_matrix):

        pass
