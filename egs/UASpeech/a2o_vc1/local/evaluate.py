#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
from collections import defaultdict
import multiprocessing as mp
import os

import numpy as np
import librosa

import torch
from tqdm import tqdm
import yaml

from s3prl_vc.utils import find_files
from s3prl_vc.evaluate.asr import load_asr_model, transcribe, calculate_measures
# from s3prl_vc.evaluate.dtw_based import calculate_mcd_f0


def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def get_number(basename):
    parts = basename.split("_")
    for i, part in enumerate(parts):
        if i == 0:
            continue
        if parts[i - 1][0] == "B" and parts[i + 1][0] == "M":
            return part


def _calculate_asv_score(file_list, gt_root, trgspk, threshold, backend, device):
    # lazy import
    if backend == "wespeaker":
        from s3prl_vc.evaluate.asv_wespeaker import load_asv_model, calculate_accept
    elif backend == "resemblyzer":
        raise NotImplementedError

    # load ASV model
    asv_model = load_asv_model(device)

    results = {}
    for i, cvt_wav_path in enumerate(tqdm(file_list)):
        basename = get_basename(cvt_wav_path)
        src_number = "_".join(basename.split("_")[1:])

        # get ground truth target wav path
        gt_wav_path = os.path.join(gt_root, trgspk + "_" + src_number + ".wav")

        # calculate acept
        results[basename] = calculate_accept(
            cvt_wav_path, gt_wav_path, asv_model, threshold
        )

    return results, 100.0 * float(np.mean(np.array(list(results.values()))))


def _calculate_asr_score(model, device, file_list, groundtruths):
    keys = ["hits", "substitutions", "deletions", "insertions"]
    ers = {}
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for i, cvt_wav_path in enumerate(tqdm(file_list)):
        basename = get_basename(cvt_wav_path)
        number = get_number(basename)
        if number.startswith("UW"):
            number =  "B2_" + number
        groundtruth = groundtruths[number]

        # load waveform
        wav, _ = librosa.load(cvt_wav_path, sr=16000)

        # trascribe
        transcription = transcribe(model, device, wav)

        # error calculation
        c_result, w_result, norm_groundtruth, norm_transcription = calculate_measures(
            groundtruth, transcription
        )

        ers[basename] = [
            c_result["cer"] * 100.0,
            w_result["wer"] * 100.0,
            norm_transcription,
            norm_groundtruth,
        ]

        for k in keys:
            c_results[k] += c_result[k]
            w_results[k] += w_result[k]

    # calculate over whole set
    def er(r):
        return (
            float(r["substitutions"] + r["deletions"] + r["insertions"])
            / float(r["substitutions"] + r["deletions"] + r["hits"])
            * 100.0
        )

    cer = er(c_results)
    wer = er(w_results)

    return ers, cer, wer


def _calculate_mcd_f0(file_list, gt_root, trgspk, f0min, f0max, results):
    for i, cvt_wav_path in enumerate(file_list):
        basename = get_basename(cvt_wav_path)
        number = get_number(basename)

        # get ground truth target wav path
        gt_wav_path = os.path.join(gt_root, number + ".wav")

        # read both converted and ground truth wav
        cvt_wav, cvt_fs = librosa.load(cvt_wav_path, sr=None)
        gt_wav, gt_fs = librosa.load(gt_wav_path, sr=None)
        assert cvt_fs == gt_fs

        # calculate MCD, F0RMSE, F0CORR and DDUR
        mcd, f0rmse, f0corr, ddur = calculate_mcd_f0(
            cvt_wav, gt_wav, gt_fs, f0min, f0max
        )

        results.append([basename, mcd, f0rmse, f0corr, ddur])


def get_parser():
    parser = argparse.ArgumentParser(description="objective evaluation script.")
    parser.add_argument(
        "--wavdir", required=True, type=str, help="directory for converted waveforms"
    )
    parser.add_argument("--trgspk", required=True, type=str, help="target speaker")
    parser.add_argument(
        "--data_root", type=str, default="./data", help="directory of data"
    )
    parser.add_argument(
        "--f0_path", required=True, type=str, help="yaml file storing f0 ranges"
    )
    parser.add_argument(
        "--threshold_path",
        required=True,
        type=str,
        help="yaml file storing thresholds for converion accuracy",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="data split. choices: train, test",
    )
    parser.add_argument(
        "--asv_backend",
        type=str,
        default="wespeaker",
        choices=["wespeaker", "resemblyzer"],
        help="speaker embedding model backend",
    )
    parser.add_argument(
        "--n_jobs", default=10, type=int, help="number of parallel jobs"
    )
    return parser


def main():
    args = get_parser().parse_args()

    trgspk = args.trgspk
    gt_root = os.path.join(args.data_root, "wav_trim", args.trgspk)
    transcription_path = os.path.join(args.data_root, "speaker_wordlist.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load f0min and f0 max
    with open(args.f0_path, "r") as f:
        f0_all = yaml.load(f, Loader=yaml.FullLoader)
    f0min = f0_all[trgspk]["minf0"]
    f0max = f0_all[trgspk]["maxf0"]

    # load ground truth transcriptions
    with open(transcription_path, "r") as f:
        lines = f.read().splitlines()
    groundtruths = {line.split(",")[1]: line.split(",")[0] for line in lines[1:]}

    # find converted files
    if args.split == "train":
        converted_files = sorted(find_files(args.wavdir, query="*B[13]*.wav"))
    elif args.split == "test":
        converted_files = sorted(find_files(args.wavdir, query="*B2*.wav"))
    print("number of utterances = {}".format(len(converted_files)))

    # load threshold; if does not exist, calculate
    if os.path.exists(args.threshold_path):
        with open(args.threshold_path, "r") as f:
            equal_error_rate, threshold = yaml.load(f, Loader=yaml.FullLoader)
    else:
        equal_error_rate, threshold = calculate_threshold(gt_root, device)
        with open(args.threshold_path, "w") as f:
            yaml.safe_dump([equal_error_rate, threshold], f)
    print(f"[INFO]: Equal error rate: {equal_error_rate}")
    print(f"[INFO]: Threshold: {threshold}")

    ##############################

    print("Calculating ASV-based score...")
    accept_results, accept_rate = _calculate_asv_score(
        converted_files, gt_root, args.trgspk, threshold, args.asv_backend, device
    )
    print("Accept rate: {:.2f}".format(accept_rate))

    ##############################

    print("Calculating ASR-based score...")
    # load ASR model
    asr_model = load_asr_model(device)

    # calculate error rates
    ers, cer, wer = _calculate_asr_score(
        asr_model, device, converted_files, groundtruths
    )

    ##############################

    spk_wise_cer_dict = defaultdict(list)
    spk_wise_wer_dict = defaultdict(list)
    spk_wise_asv_dict = defaultdict(list)
    for k, v in ers.items():
        srcspk = k.split("_")[0]
        cer = v[0]
        wer = v[1]
        asv = accept_results[k]
        spk_wise_cer_dict[srcspk].append(cer)
        spk_wise_wer_dict[srcspk].append(wer)
        spk_wise_asv_dict[srcspk].append(asv)
        print("{} {:.1f} {:.1f} {} | {} {}".format(k, *v, asv))
    spk_wise_cer = {
        spk: sum(cer_list) / len(cer_list)
        for spk, cer_list in spk_wise_cer_dict.items()
    }
    spk_wise_wer = {
        spk: sum(wer_list) / len(wer_list)
        for spk, wer_list in spk_wise_wer_dict.items()
    }
    spk_wise_asv = {
        spk: sum(asv_list) / len(asv_list)
        for spk, asv_list in spk_wise_asv_dict.items()
    }
    for spk in sorted(spk_wise_cer.keys()):
        print(
            "{}: CER:{:.2f}, WER:{:.2f}, ASV:{:.2f}".format(
                spk, spk_wise_cer[spk], spk_wise_wer[spk], spk_wise_asv[spk][0][0]
            )
        )


if __name__ == "__main__":
    main()
