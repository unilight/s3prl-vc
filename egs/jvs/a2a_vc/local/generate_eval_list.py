#!/usr/bin/env python3

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import os
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_csv", type=str, required=True, help="original csv file")
    parser.add_argument("--out", type=str, required=True, help="output file")
    args = parser.parse_args()

    # read original csv
    with open(args.original_csv, "r") as f:
        reader = csv.DictReader(f)

        # write
        with open(args.out, "w") as f:
            for row in reader:
                f.write(f"{row['sample_id']} {row['src_wav_path']} {row['ref_wav_path']}\n")
