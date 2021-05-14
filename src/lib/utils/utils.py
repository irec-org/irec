from concurrent.futures import ProcessPoolExecutor
import math
import collections
from tqdm import tqdm
import os
import multiprocessing
import numpy as np
from pathlib import Path
from . import Parameterizable

def dict_to_list_gen(d):
    for k, v in zip(d.keys(), d.values()):
        if v == None:
            continue
        yield k
        yield v

def dict_to_list(d):
    return list(dict_to_list_gen(d))

def value_to_str(value):
    if isinstance(value,dict):
        return f"{{{dict_to_str(value)}}}"
    elif isinstance(value, list):
        return f"{join_strings(list(map(lambda x: value_to_str(x), value)))}"
    elif isinstance(value, Parameterizable.Parameterizable):
        return value.get_id()
    else:
        return f"{str(value).replace('/','|')}"

def key_value_to_str(key,value):
    return f"{key}:{value_to_str(value)}"

def join_strings(strings,num_bars=0):
    return "/".join(strings[:num_bars])+("/" if num_bars and len(strings[num_bars:]) != 0 else "") +",".join(strings[num_bars:])

def dict_to_str(dictionary,num_bars=0):
    strings = []
    for key, value in dictionary.items():
        strings.append(key_value_to_str(key,value))

    return join_strings(strings,num_bars=num_bars)

def print_dict(dictionary,prefix=''):
    for key, value in dictionary.items():
        if isinstance(value,dict):
            print(f"{prefix}{key}:")
            print_dict(value,prefix+'\t')
        else:
            print(f"{prefix}{key}: {value}")

def run_parallel(func, args, use_tqdm=True):
    executor = ProcessPoolExecutor()
    num_args = len(args)
    chunksize = math.ceil(num_args/multiprocessing.cpu_count())
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

def repair_path_name(path):
    length = len(path)
    new_file_name = []
    for i in path.split('/'):
        if len(i) > 255:
            lists = [i[j:j+255] for j in range(0, len(i), 255)]
            new_file_name.append('/'.join(lists))
        else:
            new_file_name.append(i)
    return '/'.join(new_file_name)
