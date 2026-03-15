#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import librosa
import os
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description="get duration stats.")
    parser.add_argument(
        "--scp", required=True, type=str, help="wav scp"
    )
    return parser


def main():
    args = get_parser().parse_args()

    with open(args.scp, "r") as f:
        lines = f.read().splitlines()

    ds = []
    for line in tqdm(lines):
        ds.append(librosa.get_duration(path=line.split(" ")[1]))
    print(max(ds))


if __name__ == "__main__":
    main()
