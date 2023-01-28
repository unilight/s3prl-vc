#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Calculate statistics of feature files."""

import argparse
import logging
import os

import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from s3prl_vc.utils import write_hdf5
from s3prl_vc.datasets.datasets import AudioSCPMelDataset


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean and variance of dumped raw features "
            "(See detail in bin/compute_statistics.py)."
        )
    )
    parser.add_argument(
        "--scp",
        default=None,
        type=str,
        required=True,
        help=(
            "kaldi-style feats.scp file. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        required=True,
        help=(
            "directory to save statistics. if not provided, "
            "stats will be saved in the above root directory. (default=None)"
        ),
    )
    parser.add_argument("--f0", action="store_true", help="calculate f0 statistics")
    parser.add_argument(
        "--f0_path", default=None, type=str, help="yaml file storing f0 ranges"
    )
    parser.add_argument(
        "--spk", default=None, type=str, help="speaker (for getting the f0 range)"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # load config
    if args.f0:
        with open(args.f0_path) as f:
            f0_config = yaml.load(f, Loader=yaml.Loader)
            f0min = f0_config[args.spk]["f0min"]
            f0max = f0_config[args.spk]["f0max"]

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    if args.f0:
        dataset = AudioSCPMelDataset(
            args.scp,
            config,
            extract_f0=config.get("use_f0", False),
            f0_extractor=config.get("f0_extractor", "world"),
            f0_min=f0min,
            f0_max=f0max,
            log_f0=config.get("log_f0", True),
        )
    else:
        dataset = AudioSCPMelDataset(
            args.scp,
            config,
        )
    logging.info(f"The number of files = {len(dataset)}.")

    # calculate statistics
    scaler = StandardScaler()
    for items in tqdm(dataset):
        mel = items["mel"]
        scaler.partial_fit(mel)

    # write statistics
    write_hdf5(
        os.path.join(args.dumpdir, "stats.h5"),
        "mean",
        scaler.mean_.astype(np.float32),
    )
    write_hdf5(
        os.path.join(args.dumpdir, "stats.h5"),
        "scale",
        scaler.scale_.astype(np.float32),
    )

    if args.f0:
        scaler = StandardScaler()
        minmaxscaler = MinMaxScaler()
        for items in tqdm(dataset):
            f0 = items["f0"]
            f0 = f0[f0 > 0]
            scaler.partial_fit(f0.reshape([-1, 1]))
            minmaxscaler.partial_fit(f0.reshape([-1, 1]))

        # write statistics
        write_hdf5(
            os.path.join(args.dumpdir, "stats.h5"),
            "lf0_mean",
            scaler.mean_.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, "stats.h5"),
            "lf0_scale",
            scaler.scale_.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, "stats.h5"),
            "lf0_max",
            minmaxscaler.data_max_.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, "stats.h5"),
            "lf0_min",
            minmaxscaler.data_min_.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, "stats.h5"),
            "f0_max",
            f0max,
        )
        write_hdf5(
            os.path.join(args.dumpdir, "stats.h5"),
            "f0_min",
            f0min,
        )


if __name__ == "__main__":
    main()
