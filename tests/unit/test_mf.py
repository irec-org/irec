from irec.mf.MF import MF
from irec.mf.NMF import NMF

def test_create_value_functions():
    assert isinstance(NMF(), MF)
