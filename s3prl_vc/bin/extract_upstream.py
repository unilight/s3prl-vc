#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Extract upstream features."""

import argparse
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
import yaml

from tqdm import tqdm

from s3prl.nn import Featurizer

import s3prl_vc.models
from s3prl_vc.upstream.interface import get_upstream
from s3prl_vc.datasets.datasets import AudioSCPMelDataset
from s3prl_vc.utils import read_hdf5, write_hdf5
from s3prl_vc.utils.data import pad_list
from s3prl_vc.utils.plot import plot_generated_and_ref_2d, plot_1d
from s3prl_vc.vocoder import Vocoder


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=("Decode with trained model " "(See detail in bin/decode.py).")
    )
    parser.add_argument(
        "--scp",
        type=str,
        default=None,
        help=("kaldi-style wav.scp file. "),
    )
    parser.add_argument(
        "--wavdir",
        default=None,
        type=str,
        help=(
            "directory including input wav files. you need to specify either scp or wavdir."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated stuff.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded (for featurizer).",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help=(
            "yaml format configuration file. if not explicitly provided, "
            "it will be searched in the checkpoint directory. (default=None)"
        ),
    )
    parser.add_argument(
        "--feat_type",
        type=str,
        default="feats",
        help=("feature type. this is used as key name to read h5 featyre files. "),
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

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # check arguments
    if (args.scp is not None and args.wavdir is not None) or (
        args.scp is None and args.wavdir is None
    ):
        raise ValueError("Please specify either --wavdir or --scp.")

    # get dataset
    if args.scp is not None:
        dataset = AudioSCPMelDataset(
            args.scp,
            config,
            return_utt_id=True,
        )
    else:
        dataset = AudioMelDataset(
            args.wavdir,
            config,
            return_utt_id=True,
        )

    logging.info(f"The number of files to be extracted = {len(dataset)}.")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    # define upstream model
    upstream_model = get_upstream(config["upstream"]).to(device)
    upstream_model.eval()
    upstream_featurizer = Featurizer(upstream_model).to(device)
    upstream_featurizer.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["featurizer"]
    )
    upstream_featurizer.eval()
    logging.info(f"Loaded model parameters from {args.checkpoint}.")

    # start generation
    with torch.no_grad():
        for items in tqdm(dataset):
            utt_id = items["utt_id"]
            x = items["audio"]
            xs = torch.from_numpy(x).unsqueeze(0).float().to(device)
            ilens = torch.LongTensor([x.shape[0]]).to(device)

            start_time = time.time()
            all_hs, all_hlens = upstream_model(xs, ilens)
            hs, _ = upstream_featurizer(all_hs, all_hlens)
            h = hs[0]
            logging.info(
                "inference speed = %.1f frames / sec."
                % (int(h.size(0)) / (time.time() - start_time))
            )

            # write feats
            if not os.path.exists(os.path.join(config["outdir"], args.feat_type)):
                os.makedirs(
                    os.path.join(config["outdir"], args.feat_type), exist_ok=True
                )

            write_hdf5(
                config["outdir"] + f"/{args.feat_type}/{utt_id}.h5",
                args.feat_type,
                h.cpu().numpy().astype(np.float32),
            )


if __name__ == "__main__":
    main()
