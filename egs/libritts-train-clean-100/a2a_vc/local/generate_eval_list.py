#!/usr/bin/env python3

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import os

SRCSPKS = ["SEF1", "SEF2", "SEM1", "SEM2"]
TRGSPKS = ["TEF1", "TEF2", "TEM1", "TEM2"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, required=True, help="root directory of wav files")
    parser.add_argument("--train_list", type=str, required=True, help="list of training data to be used")
    parser.add_argument("--eval_list", type=str, required=True, help="list of evaluation data to be used")
    parser.add_argument("--num_ref_samples", type=int, required=True, help="number of samples to be used for reference")
    parser.add_argument("--out", type=str, required=True, help="output file")
    args = parser.parse_args()

    # read train list
    with open(args.train_list) as f:
        train_list = f.read().splitlines()

    with open(args.eval_list) as f:
        eval_list = f.read().splitlines()

    with open(args.out, "w") as f:
        for srcspk in SRCSPKS:
            for trgspk in TRGSPKS:
                for eval_utt_id in eval_list:
                    utt_id = f"{srcspk}_{trgspk}_{eval_utt_id}"
                    srcwav = os.path.join(args.rootdir, srcspk, f"{eval_utt_id}.wav")
                    f.write(f"{utt_id} {srcwav} ")

                    refwavs = []
                    for train_utt_id in train_list:
                        refwav = os.path.join(args.rootdir, trgspk, f"{train_utt_id}.wav")
                        if os.path.isfile(refwav):
                            refwavs.append(refwav)
                            if len(refwavs) >= args.num_ref_samples:
                                break
                    f.write(" ".join(refwavs))

                    f.write("\n")