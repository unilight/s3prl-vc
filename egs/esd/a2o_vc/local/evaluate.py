#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 Wen-Chin Huang
#  MIT License (https://opensource.org/licenses/MIT)

import argparse
import multiprocessing as mp
import os

import numpy as np
import librosa

import torch
from tqdm import tqdm
import yaml

from s3prl_vc.utils import find_files
from s3prl_vc.evaluate.dtw_based import calculate_mcd_f0
from s3prl_vc.evaluate.asr import load_asr_model, transcribe, calculate_measures

MAPPING = {"Neutral": 0, "Angry": 1, "Happy": 2, "Sad": 3, "Surprise": 4}
ENCODINGS = {
    "0001":	"iso-8859-1",
    "0002":	"iso-8859-1",
    "0003":	"utf-16le",
    "0004":	"iso-8859-1",
    "0005":	"iso-8859-1",
    "0006":	"utf-16le",
    "0007":	"utf-16le",
    "0008":	"utf-16le",
    "0009":	"iso-8859-1",
    "0010":	"utf-16le",
    "0011":	"us-ascii",
    "0012":	"utf-16le",
    "0013":	"utf-16le",
    "0014":	"utf-16le",
    "0015":	"us-ascii",
    "0016":	"iso-8859-1",
    "0017":	"iso-8859-1",
    "0018":	"utf-16le",
    "0019":	"utf-16le",
    "0020":	"us-ascii",
}

def get_original_number(number, spk, emo):
    return spk + "_" + f"{(int(number) + MAPPING[emo] * 350):06}"


def get_basename(path):
    return os.path.splitext(os.path.split(path)[-1])[0]


def _calculate_asr_score(model, device, file_list, groundtruths):
    keys = ["hits", "substitutions", "deletions", "insertions"]
    ers = {}
    c_results = {k: 0 for k in keys}
    w_results = {k: 0 for k in keys}

    for i, cvt_wav_path in enumerate(tqdm(file_list)):
        basename = get_basename(cvt_wav_path)
        groundtruth = groundtruths[basename]

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


def _calculate_mcd_f0(file_list, gt_files, srcemo, trgspk, trgemo, f0_all, results, scenario):
    for i, cvt_wav_path in enumerate(file_list):
        basename = get_basename(cvt_wav_path) # 1
        if scenario == "spevc":
            original_number = get_original_number(basename, trgspk, trgemo) # 1 -> 0013_000351
            f0min = f0_all[trgspk+"_"+trgemo]["f0min"]
            f0max = f0_all[trgspk+"_"+trgemo]["f0max"]
        elif scenario == "epsvc":
            original_number = get_original_number(basename, trgspk, srcemo) # 1 -> 0013_000351
            f0min = f0_all[trgspk+"_"+srcemo]["f0min"]
            f0max = f0_all[trgspk+"_"+srcemo]["f0max"]
        
        # get ground truth target wav path
        candidates = [f for f in gt_files if get_basename(f) == original_number]
        assert len(candidates) == 1
        gt_wav_path = candidates[0]

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
    parser.add_argument("--srcemo", required=True, type=str, help="source emotion")
    parser.add_argument("--trgspk", required=True, type=str, help="target speaker")
    parser.add_argument("--trgemo", required=True, type=str, help="target emotion")
    parser.add_argument(
        "--data_root", type=str, default="./data", help="directory of data"
    )
    parser.add_argument(
        "--f0_path", required=True, type=str, help="yaml file storing f0 ranges"
    )
    parser.add_argument(
        "--scenario", required=True, type=str, help="conversion scenario", choices=["spevc", "epsvc"]
    )
    parser.add_argument(
        "--n_jobs", default=10, type=int, help="number of parallel jobs"
    )
    return parser


def main():
    args = get_parser().parse_args()

    srcemo = args.srcemo
    trgspk = args.trgspk
    trgemo = args.trgemo
    gt_root = args.data_root
    transcription_path = os.path.join(args.data_root, trgspk, trgspk + ".txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load f0min and f0 max
    with open(args.f0_path, 'r') as f:
        f0_all = yaml.load(f, Loader=yaml.FullLoader)
    

    # load ground truth transcriptions
    with open(transcription_path, "r", encoding=ENCODINGS[trgspk]) as f:
        lines = [line for line in f.read().splitlines() if len(line) > 0]
    groundtruths = {}
    for line in lines:
        num_tab = line.count("\t")
        if num_tab == 2:
            number, sentence, emotion = line.split("\t")
        elif num_tab == 1:
            number_sentence, emotion = line.split("\t")
            number = number_sentence.split(" ")[0]
            sentence = " ".join(number_sentence.split(" ")[1:])
        elif num_tab == 0:
            number = line.split(" ")[0]
            emotion = line.split(" ")[-1]
            sentence = " ".join(number_sentence.split(" ")[1:-1])
        else:
            print("wtf is happening")
            exit(1)
        if emotion == trgemo:
            real_number = str(int(number.split("_")[1]) % 350)
            groundtruths[real_number] = sentence

    # find converted files
    converted_files = sorted(find_files(args.wavdir, query="*.wav"))
    print("number of utterances = {}".format(len(converted_files)))

    # find ground truth files
    gt_files = sorted(find_files(os.path.join(gt_root, trgspk), query="*.wav"))

    ##############################

    print("Calculating ASR-based score...")
    # load ASR model
    asr_model = load_asr_model(device)

    # calculate error rates
    ers, cer, wer = _calculate_asr_score(
        asr_model, device, converted_files, groundtruths
    )

    ##############################

    print("Calculating MCD and f0-related scores...")
    # Get and divide list
    file_lists = np.array_split(converted_files, args.n_jobs)
    file_lists = [f_list.tolist() for f_list in file_lists]

    # multi processing
    with mp.Manager() as manager:
        results = manager.list()
        processes = []
        for f in file_lists:
            p = mp.Process(
                target=_calculate_mcd_f0,
                args=(f, gt_files, args.srcemo, trgspk, trgemo, f0_all, results, args.scenario),
            )
            p.start()
            processes.append(p)

        # wait for all process
        for p in processes:
            p.join()

        results = sorted(results, key=lambda x: x[0])
        results = [result + ers[result[0]] for result in results]

    # utterance wise result
    for result in results:
        print(
            "{} {:.2f} {:.2f} {:.2f} {:.2f} {:.1f} {:.1f} \t{} | {}\n".format(*result)
        )

    # average result
    mMCD = np.mean(np.array([result[1] for result in results]))
    mf0RMSE = np.mean(np.array([result[2] for result in results]))
    mf0CORR = np.mean(np.array([result[3] for result in results]))
    mDDUR = np.mean(np.array([result[4] for result in results]))
    mCER = cer
    mWER = wer

    print(
        "Mean MCD, f0RMSE, f0CORR, DDUR, CER, WER: {:.2f} {:.2f} {:.3f} {:.3f} {:.1f} {:.1f}".format(
            mMCD, mf0RMSE, mf0CORR, mDDUR, mCER, mWER
        )
    )


if __name__ == "__main__":
    main()
