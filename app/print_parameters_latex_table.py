#!/usr/bin/python3
from os.path import dirname, realpath
import yaml
import os
import argparse

from irec.connector import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_loaders", nargs="*")
parser.add_argument("--agents", nargs="*")
args = parser.parse_args()

# settings = utils.load_settings(dirname(realpath(__file__)))
dataset_agents_parameters = yaml.load(open("./settings/dataset_agents.yaml"),Loader=yaml.SafeLoader)
# settings = yaml.load("./settings/",Loader=yaml.SafeLoader)
settings = utils.load_settings(dirname(realpath(__file__)))

parameters_agents_symbols = {
    "Popular": {},
    "Random": {},
    "MostPopular": {},
    "UCB": {'c':'c'},
    "ThompsonSampling": {'alpha_0':r'$\alpha_0$','beta_0':r'$\beta_0$'},
    "EGreedy": {'epsilon':r'$\varepsilon$'},
    "LinearUCB": {
          'alpha': r'$\alpha$',
          'item_var': r'$\sigma^2_{q}$',
          'iterations': r'$T$',
          'num_lat': r'num\_lat',
          'stop_criteria': r'stop\_criteria',
          'user_var': r'$q^2_{p}$',
          'var': r'$\sigma^2$',
          },
    "GLM_UCB": {
          'c': 'c',
          'item_var': r'$\sigma^2_{q}$',
          'iterations': r'$T$',
          'num_lat': r'num\_lat',
          'stop_criteria': r'stop\_criteria',
          'user_var': r'$q^2_{p}$',
          'var': r'$\sigma^2$',
        },
    "NICF": {
          "batch":'batch',
          "clip_param":r'clip\_param',
          "dropout_rate":r'dropout\_rate',
          "gamma":'gamma',
          "inner_epoch":r'inner\_epoch',
          "latent_factor":r'latent\_factor',
          "learning_rate":r'learning\_rate',
          "num_blocks":r'num\_blocks',
          "num_heads":r'num\_heads',
          "restore_model":r'restore\_model',
          "rnn_layer":r'rnn\_layer',
          "time_step":r'time\_step',
          "training_epoch":r'training\_epoch',
        },
    "PTS": {
        "num_lat":r'num\_lat',
        "num_particles":'D',
        "var":r'$\sigma^2$',
        "var_u":r'$\sigma^2_U$',
        "var_v":r'$\sigma^2_V$',
        },
    "ICTRTS": {
          'num_lat':'num\_lat',
          'num_particles':'$B$',
        },
    "cluster_bandit": {
          'B': 'B',
          'C': 'C',
          'D': 'C',
          'num_clusters': r'num\_clusters',
          'num_lat': r'num\_lat',
          },
    "WSPB": {
          'alpha':r'$\alpha$' ,
          'num_lat': r'num\_lat' ,
          },
}

utils.print_parameters_latex_table(
    dataset_agents_parameters, args.agents, args.dataset_loaders,parameters_agents_symbols, settings
)
