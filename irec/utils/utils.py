from concurrent.futures import ProcessPoolExecutor
import math
from tqdm import tqdm
import multiprocessing
import numpy as np


class TupleNonRedundantList:
    def __init__(self, redundant_key, non_redundant_list) -> None:
        self.non_redundant_list = non_redundant_list
        self.redundant_key = redundant_key
        pass

    def __delitem__(self, key):
        # self.non_redundant_list[1].__delattr__(key)
        raise NotImplementedError

    def __getitem__(self, key):
        return (self.redundant_key, self.non_redundant_list.__getattribute__(key))

    def __setitem__(self, key, value):
        raise NotImplementedError
        # self.non_redundant_list[1].__setattr__(key, value)


def dict_to_list_gen(d):
    for k, v in zip(d.keys(), d.values()):
        if v is None:
            continue
        yield k
        yield v


def dict_to_list(d):
    return list(dict_to_list_gen(d))


def value_to_str(value):
    if isinstance(value, dict):
        return f"{{{dict_to_str(value)}}}"
    elif isinstance(value, list):
        return f"{join_strings(list(map(lambda x: value_to_str(x), value)))}"
    else:
        return f"{str(value).replace('/','|')}"


def key_value_to_str(key, value):
    return f"{key}:{value_to_str(value)}"


def join_strings(strings, num_bars=0):
    return (
        "/".join(strings[:num_bars])
        + ("/" if num_bars and len(strings[num_bars:]) != 0 else "")
        + ",".join(strings[num_bars:])
    )


def dict_to_str(dictionary, num_bars=0):
    strings = []
    for key, value in dictionary.items():
        strings.append(key_value_to_str(key, value))

    return join_strings(strings, num_bars=num_bars)


def print_dict(dictionary, prefix=""):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict(value, prefix + "\t")
        else:
            print(f"{prefix}{key}: {value}")


def run_parallel(func, args, use_tqdm=True):
    executor = ProcessPoolExecutor()
    num_args = len(args)
    chunksize = math.ceil(num_args / multiprocessing.cpu_count())
    if use_tqdm:
        ff = tqdm
    else:
        ff = lambda x, *y, **z: x
    results = [
        i
        for i in ff(
            executor.map(func, *list(zip(*args)), chunksize=chunksize), total=num_args
        )
    ]
    return results


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def repair_path_name(path):
    length = len(path)
    new_file_name = []
    for i in path.split("/"):
        if len(i) > 255:
            lists = [i[j : j + 255] for j in range(0, len(i), 255)]
            new_file_name.append("/".join(lists))
        else:
            new_file_name.append(i)
    return "/".join(new_file_name)
