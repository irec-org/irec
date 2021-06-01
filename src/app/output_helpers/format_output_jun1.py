#!/usr/bin/python3
import pandas as pd
from tabulate import tabulate
from io import StringIO
# pd.read
f=open('./output_correlations_new_jun1.txt')
string = [line for line in f  if 'Correlation:' in line]
string = '\n'.join(string)

df = pd.read_csv(StringIO(string), delimiter = " ",header=None)
df = df[[1,3,5,7]]
df[5] = pd.to_numeric(df[5])
df[7] = pd.to_numeric(df[7])
df= df.replace(69878,'MovieLens 10M')
df= df.replace(15400,'Yahoo Music')
df= df.replace(53423,'Good Books')
df=  df.sort_values(by=[1,3])
df = df.reset_index(drop=True)
# pd.to_numeric(df)
df.columns = ['Dataset','Method','Pop. Corr.','Ent. Corr.']
# print(df)
# print(df.head())
print(tabulate(df, headers='keys', tablefmt='psql'))
