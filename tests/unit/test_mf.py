from irec.matrix_factorization.MF import MF
from irec.matrix_factorization.NMF import NMF

def test_create_value_functions():
    assert isinstance(NMF(), MF)
