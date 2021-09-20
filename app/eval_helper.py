#!/bin/python3
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("-m", nargs="*")
parser.add_argument("-b", nargs="*")
parser.add_argument("-e", nargs="*")
# args = parser.parse_args()
args, unknownargs = parser.parse_known_args()
# print(unknownargs)

for b in args.b:
    for m in args.m:
        for e in args.e:
            os.system(
                f'python3 eval.py -b "{b}" -m "{m}" -e "{e}" {" ".join(unknownargs)}'
            )
