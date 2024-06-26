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

from tqdm import tqdm

from s3prl_vc.utils import write_hdf5
from s3prl_vc.datasets.datasets import AudioSCPMelDataset
from s3prl_vc.utils.speaker_embedding_resemblyzer import (
    load_asv_model,
    get_embedding,
)


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute speaker embedding " "(See detail in bin/extract_spemb.py)."
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

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    dataset = AudioSCPMelDataset(
        config,
        args.scp,
        return_utt_id=True,
        return_wavpath=True,
    )
    logging.info(f"The number of files = {len(dataset)}.")

    # load speaker encoder
    spk_emb_model = load_asv_model()

    # calculate speaker embedding
    for items in tqdm(dataset):
        utt_id = items["utt_id"]
        wavpath = items["wavpath"]
        spk_emb = get_embedding(wavpath, spk_emb_model)

        # write to file
        write_hdf5(
            os.path.join(args.dumpdir, utt_id + ".h5"),
            "spemb",
            spk_emb.astype(np.float32),
        )


if __name__ == "__main__":
    main()
