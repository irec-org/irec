from os.path import dirname, realpath, sep, pardir
import pickle
import os
import sys
import json
import utils
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "lib")
from lib.utils.DirectoryDependent import DirectoryDependent
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
import pyspark.sql.types as T
from pyspark import SparkConf
import numpy as np
import argparse
import pandas as pd

def spark_config():

    config = SparkConf() \
        .set('spark.executor.memory', '1G') \
        .set('spark.driver.memory', '1G')

    spark = SparkSession \
        .builder \
        .master('local[*]') \
        .appName('data-volumetric') \
        .config(conf=config) \
        .getOrCreate()

    return spark

def fff(number):
    return str(number)[:str(number).find('.')+4]

def stages_accuracy(df, method, dataset, values):
    
    df = df.withColumn("hit", F.when(F.col("reward") >= 4, 1).otherwise(0))
    result = df.select("uid").distinct()
    
    columns = []
    
    for idx in range(0, len(values)-1):
        
        nb_items = values[idx+1] - values[idx]
        
        df_aux = df.filter((F.col("user_trial") > values[idx]) & (F.col("user_trial") <= int(values[idx+1])))\
                   .groupBy("uid").agg((F.sum(F.col("hit"))/nb_items).alias(f"hits_s{idx}"))
        result = result.join(df_aux, on="uid", how="left")
        columns.append(f"hits_s{idx}")
          
    
    result = result\
                    .join(
                        df.filter(F.col("user_trial") <= values[-1])\
                          .groupBy("uid").agg((F.sum(F.col("hit"))/values[-1]).alias("hits_total")),
                        on="uid", how="left"
                        )
    columns.append("hits_total")
    
    return result.select(*[F.mean(c).alias(c) for c in columns])\
                 .withColumn("method", F.lit(method))\
                 .withColumn("dataset", F.lit(dataset))

parser = argparse.ArgumentParser()
parser.add_argument('-m', nargs='*')
parser.add_argument('-b', nargs='*')
args = parser.parse_args()

spark = spark_config()
dd = DirectoryDependent()
input_path = f'{dd.DIRS["export"]}/'

datasets = args.b
methods = args.m

values = [0, 5, 10, 15, 20, 50, 100]
df_result = pd.DataFrame()

for dataset in datasets: 
    for m in methods:
        path = f"{input_path}d_{dataset}_{m}.parquet"
        df = spark.read.option("inferSchema", "true")\
                       .load(path)\
                       .withColumn("trial", F.monotonically_increasing_id())
        w = Window.partitionBy("uid").orderBy("trial")
        df = df.withColumn("user_trial", F.row_number().over(w))
        df_result = df_result.append(stages_accuracy(df, m, dataset, values=values).select("*").toPandas())

columns = ["hits_s0","hits_s1","hits_s2","hits_s3","hits_s4","hits_s5","hits_total"]

df_analysis = pd.DataFrame()
for d in datasets:
	df = df_result.loc[df_result["dataset"] == d]
	for col in columns:
	    maxv = df[col].max()
	    df[col] = df[col].apply(lambda x: fff(float(x)) if x != maxv else "\\textbf{\cellcolor{green!25}"+fff(float(x)) +"}")
	df_analysis = df_analysis.append(df)

latex_table = r"""
\begin{table*}[!htb]
\footnotesize
%\footnotesize
\begin{adjustbox}{center}
\setlength\tabcolsep{2.5pt}
\begin{tabular}{|c|cccccc|c|}
\hline
\rowcolor{StrongGray}
Dataset & \multicolumn{7}{c|}{mydf}\\
\hline
\hline
\rowcolor{Gray}
Measure 
& \multicolumn{6}{c|}{Precision at each stage} & \multirow{2}{*}\textbf{Total} \\
\hline
\rowcolor{Gray}
T & 1-5 & 6-10 &  11-15 &  16-20 &  21-50 &  51-100 & 1-100 \\
\hline
\hline

method & v1 & v2 & v3 & v4 & v5 & v6 & v7\\\hline

\end{tabular}
\end{adjustbox}
\end{table*}
"""

num_datasets = len(datasets)
num_methods = len(methods)

latex_table = latex_table.replace("cccccc|c|", "cccccc|c|" * num_datasets)
latex_table = latex_table.replace("& \multicolumn{7}{c|}{mydf}", " & \multicolumn{7}{c|}{mydf}" * num_datasets)
latex_table = latex_table.replace("& \multicolumn{6}{c|}{Precision at each stage} & \multirow{2}{*}\\textbf{Total}", 
                                 " & \multicolumn{6}{c|}{Precision at each stage} & \multirow{2}{*}\\textbf{Total}" * num_datasets)
latex_table = latex_table.replace("& 1-5 & 6-10 &  11-15 &  16-20 &  21-50 &  51-100 & 1-100",
                                  "& 1-5 & 6-10 &  11-15 &  16-20 &  21-50 &  51-100 & 1-100\n" * num_datasets)
latex_table = latex_table.replace("method & v1 & v2 & v3 & v4 & v5 & v6 & v7\\\\\hline",
                                  "method & v1 & v2 & v3 & v4 & v5 & v6 & v7\\\\\hline\n" * num_methods)
latex_table = latex_table.replace("& v1 & v2 & v3 & v4 & v5 & v6 & v7", " & v1 & v2 & v3 & v4 & v5 & v6 & v7" * num_datasets)

columns = ["hits_s0","hits_s1","hits_s2","hits_s3","hits_s4","hits_s5","hits_total"]
for dataset in datasets: latex_table = latex_table.replace("mydf", dataset, 1)

for method in methods:
    latex_table = latex_table.replace("method", method.replace("_", "-"), 1)
    for dataset in datasets:
        values = df_analysis.loc[(df_analysis["method"] == method) & (df_analysis["dataset"] == dataset)]
        for i, col in enumerate(columns): latex_table = latex_table.replace(f"v{i+1}", values[col].values[0],1)


rtex_header = r"""	
\documentclass{article}
%%\usepackage[landscape, paperwidth=15cm, paperheight=30cm, margin=0mm]{geometry}
\usepackage{multirow}
\usepackage{color, colortbl}
\usepackage{xcolor, soul}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[brazil]{babel}
\usepackage{graphicx}
\usepackage{subcaption}  
\usepackage{adjustbox}

\definecolor{Gray}{gray}{0.9}
\definecolor{StrongGray}{gray}{0.7}

\title{Dataset Information}
\begin{document}
			"""

rtex_footer = r"""
\end{document}
"""

latex_table = rtex_header + latex_table + rtex_footer
tmp = "_".join([d for d in datasets])+"_"+"_".join([m for m in methods])+"_precision_each_stage"
open(os.path.join(DirectoryDependent().DIRS['tex'], f'table_{tmp}.tex'),
     'w+').write(latex_table)
os.system(
    f"pdflatex -output-directory=\"{DirectoryDependent().DIRS['pdf']}\" \"{os.path.join(DirectoryDependent().DIRS['tex'],f'table_{tmp}.tex')}\""
)
