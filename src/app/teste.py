from util import DatasetFormatter
import pandas as pd

df_cons = pd.read_csv('testSet_ML-1M_120_bin=False_.data',
                      sep='::',
                      header=None,
                      engine='python')

uids = df_cons[0].unique()
dsf = DatasetFormatter()
dsf = dsf.load()
print((set(uids) == set(dsf.test_uids)))
