#!/usr/bin/python3
import numpy as np

base_parameters = {
    "num_lat": 20,
    # "alpha": 0.75,
    # "lambda_u": 0.75,
    "lambda_": 0.75,
    "alpha": 0.5,
    # "c": 0.75,
    # 'stop':None,
    # 'weight_method':'change',
    # "init": 'entropy',
    # 'item_var': 0.01,
    # 'iterations': 20,
    # 'stop_criteria': 0.0009,
    # 'user_var': 0.01,
    # 'var': 0.05,
}


def rrange(start, end, step):
    i = start
    while i < end:
        yield i
        i += step
    yield end


num_lats = [5, 10, 20]
alphas = [0.25, 0.5, 0.75, 1]
inits = ["entropy", "popularity", "logpopent", "rand_popularity", "random"]
lambda_us = [1.164322, 1.2315279, 1.482435, 0.5, 0.1, 0.01, 0.001]
# num_lats = [5, 10, 20]
# alphas = [0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1]
lambdas_ = [0, 0.25, 0.5, 0.75, 1]
lambdas_ = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

epsilons = rrange(0.1, 0.5, 0.1)
# num_lats_pts = [2,10,20]
num_lats_nicf = rrange(10, 50, 10)
num_particles_pts = [2, 5, 10]
cs = [0.1, 0.5, 1, 2, 4, 8]
# search = {"num_lat": num_lats, "epsilon": epsilons}
# search = {"num_lat": num_lats, "alpha": alphas}
# search = {"num_lat": num_lats, "c": cs}
search = {"lambda_": lambdas_}

import itertools
from copy import copy

parameters = []

for values in itertools.product(*list(search.values())):
    for key, value in zip(list(search.keys()), values):
        # base_parameters[key] = round(value, 2)
        base_parameters[key] = value
    parameters.append(copy(base_parameters))

import yaml

print(yaml.dump(parameters))
