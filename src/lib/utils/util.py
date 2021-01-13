from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import multiprocessing
import numpy as np
from pathlib import Path

def dict_to_list_gen(d):
    for k, v in zip(d.keys(), d.values()):
        if v == None:
            continue
        yield k
        yield v

def dict_to_list(d):
    return list(dict_to_list_gen(d))

def dict_to_str(dictionary,num_bars=0):
    strings = []
    for key, value in dictionary.items():
        if isinstance(value,dict):
            strings.append(f"{key}:{{{dict_to_str(value)}}}")
        else:
            strings.append(f"{key}:{str(value).replace('/','|')}")
    return "/".join(strings[:num_bars])+("/" if num_bars and len(strings[num_bars:]) != 0 else "") +",".join(strings[num_bars:])

def run_parallel(func, args, use_tqdm=True):
    executor = ProcessPoolExecutor()
    num_args = len(args)
    chunksize = int(num_args/multiprocessing.cpu_count())
    if use_tqdm:
        ff = tqdm
    else:
        ff = lambda x,*y,**z: x 
    results = [i for i in ff(executor.map(func,*list(zip(*args)),chunksize=chunksize),total=num_args)]
    return results

def sigmoid(x):
    return 1/(1+np.exp(-x))


def create_train_test_with_results(consumption_matrix,results,rate):
    pass

def create_path_to_file(file_name):
    Path('/'.join(file_name.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
