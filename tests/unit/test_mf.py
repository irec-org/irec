from irec.recommendation.matrix_factorization.MF import MF
from irec.recommendation.matrix_factorization.NMF import NMF

def test_create_value_functions():
    assert isinstance(NMF(), MF)
