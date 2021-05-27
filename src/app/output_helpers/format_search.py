from tabulate import tabulate
# text = 
results= [['k','Method','Users','Items','Target','Correlation']]
for line in open('output_search_normalized.txt'):
   ls = line.split(' ')
   results.append([ls[0],ls[1],ls[10],ls[11],ls[7][:-1],'%.5f'%(float(ls[8][1:-1]))])

print(tabulate(results))
