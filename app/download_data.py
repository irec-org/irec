#!/usr/bin/python3
from os.path import dirname, realpath, sep, pardir
import os
import sys

sys.path.append(dirname(dirname(realpath(__file__))))
import argparse
from irec.connector.utils import download_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', nargs="*", required=True, type=str, help='Name of dataset')
args = parser.parse_args()
print(args)
download_data(args.dataset_name)
