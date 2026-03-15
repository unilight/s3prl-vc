#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import os


SPKS = [f"{i:04}" for i in range (11, 21)]
EMOS = ["Neutral", "Angry", "Happy", "Sad", "Surprise"]


def get_parser():
    parser = argparse.ArgumentParser(description="parse objective evaluation results.")
    parser.add_argument(
        "--dir", required=True, type=str, help="directory for converted waveforms"
    )
    parser.add_argument("--trgspk", required=True, type=str, help="target speaker")
    parser.add_argument("--trgemo", required=True, type=str, help="target emotion")
    parser.add_argument(
        "--scenario", required=True, type=str, help="conversion scenario", choices=["spevc", "epsvc"]
    )
    return parser


def main():
    args = get_parser().parse_args()

    print(f"Target: {args.trgspk} {args.trgemo}")

    mcds = {}
    cers = {}
    wers = {}
    print("\t" + "\t\t\t".join(EMOS) + "\t\tAverage")
    for srcspk in SPKS:
        print(f"{srcspk}", end="\t")
        for emo in EMOS:
            p = os.path.join(args.dir, f"{srcspk}_{emo}_eval", f"{args.scenario}_evaluation.log")
            if not os.path.exists(p):
                print(f"N/A", end="\t\t\t")
                continue
            with open(p) as f:
                lines = f.read().splitlines()
            result = None
            for line in lines:
                if line.startswith("Mean MCD"):
                    result = line
                    break
            if result is not None:
                mcd = result.split(" ")[7]
                cer = result.split(" ")[-2]
                wer = result.split(" ")[-1]
                mcds[srcspk + "_" + emo] = mcd
                cers[srcspk + "_" + emo] = cer
                wers[srcspk + "_" + emo] = wer
                print(f"{mcd}/{cer}/{wer}", end="\t\t")
            else:
                print(f"N/A", end="\t\t\t")
        if all([srcspk + "_" + e in mcds for e in EMOS]) and all([srcspk + "_" + e in cers for e in EMOS]) and all([srcspk + "_" + e in wers for e in EMOS]):
            mmcd = sum([float(mcds[srcspk + "_" + e]) for e in EMOS if e != args.trgemo]) / 4
            mcer = sum([float(cers[srcspk + "_" + e]) for e in EMOS if e != args.trgemo]) / 4
            mwer = sum([float(wers[srcspk + "_" + e]) for e in EMOS if e != args.trgemo]) / 4
            print(f"{mmcd:.2f}/{mcer:.1f}/{mwer:.1f}", end="")
        else:
            print(f"N/A", end="")
        print()

    print("Average", end="\t")
    for emo in EMOS:
        if all([s + "_" + emo in mcds for s in SPKS]) and all([s + "_" + emo in cers for s in SPKS]) and all([s + "_" + emo in wers for s in SPKS]):
            mmcd = sum([float(mcds[s + "_" + emo]) for s in SPKS if s != args.trgspk]) / 9
            mcer = sum([float(cers[s + "_" + emo]) for s in SPKS if s != args.trgspk]) / 9
            mwer = sum([float(wers[s + "_" + emo]) for s in SPKS if s != args.trgspk]) / 9
            print(f"{mmcd:.2f}/{mcer:.1f}/{mwer:.1f}", end="\t\t")
        else:
            print(f"N/A", end="\t\t\t")
    # mmcd = sum([float(mcds[s + "_" + e]) for s in SPKS for e in EMOS if s != args.trgspk and e != args.trgemo]) / 36
    # mcer = sum([float(cers[s + "_" + e]) for s in SPKS for e in EMOS if s != args.trgspk and e != args.trgemo]) / 36
    # mwer = sum([float(wers[s + "_" + e]) for s in SPKS for e in EMOS if s != args.trgspk and e != args.trgemo]) / 36
    # print(f"{mmcd:.2f}/{mcer:.1f}/{mwer:.1f}", end="\t\t")
    print()

if __name__ == "__main__":
    main()
