from tabulate import tabulate
import pandas as pd
df = pd.read_csv('output.txt', delimiter=" ", header=None)
# print(df)
df = df[[0, 5, 6, 8, 9]]
df[5] = df[5].str.slice(0, -1)
df[6] = pd.to_numeric(df[6].str.slice(1, -1))
df[8] = pd.to_numeric(df[8])
df = df.sort_values(by=[8, 5, 0, 6])
df = df.reset_index(drop=True)

df.columns = ['Method', 'Target', 'Correlation', 'Users', 'Items']
# print(df)
print(tabulate(df, headers='keys', tablefmt='psql'))
# text =
# results= [['k','Method','Users','Items','Target','Correlation']]
# results =[]
# for line in open('output_search_normalized.txt'):
# ls = line.split(' ')
# results.append([ls[0],ls[1],ls[10],ls[11],ls[7][:-1],'%.5f'%(float(ls[8][1:-1]))])

# print(tabulate(results))
