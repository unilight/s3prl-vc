#!/usr/bin/env python3

# Copyright 2024 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import os

from s3prl_vc.utils import find_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, required=True, help="root directory of spemb h5 files")
    parser.add_argument("--wavscp", type=str, required=True, help="wavscp")
    parser.add_argument("--out", type=str, required=True, help="output file")
    args = parser.parse_args()

    spemb_files = sorted(find_files(args.rootdir, query="*.h5"))
    _id_to_spemb_path = {}
    for f in spemb_files:
        _id = os.path.splitext(os.path.basename(f))[0]
        _id_to_spemb_path[_id] = os.path.realpath(f)

    with open(args.wavscp, "r") as rf, open(args.out, "w") as wf:
        lines = rf.read().splitlines()
        for line in lines:
            _id, _ = line.split(" ")
            wf.write(f"{_id} {_id_to_spemb_path[_id]}\n")
