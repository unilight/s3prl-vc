#!/usr/bin/env bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# basic settings
stage=-1       # stage to start
stop_stage=100 # stage to stop
verbose=1      # verbosity level (lower is less info)
n_gpus=1       # number of gpus in training
n_jobs=16      # number of parallel jobs in feature extraction

upstream=ppg_sxliu
conf=conf/taco2_ar_resem.yaml

# dataset configuration
db_root=/data/group1/z44476r/Corpora/l2-arctic

# pretrained model related
pretrained_model=           # available pretrained models: m_ailabs.judy.vtn_tts_pt

# training related setting
tag=""     # tag for directory to save model
resume=""  # checkpoint path to resume training
           # (e.g. <path>/<to>/checkpoint-10000steps.pkl)
           
# decoding related setting
outdir=                     # In case not evaluation not executed together with decoding & synthesis stage
model=                      # VC Model checkpoint for decoding. If not specified, automatically set to the latest checkpoint 
checkpoint=""               # checkpoint path to be used for decoding
                            # if not provided, the latest one will be used
                            # (e.g. <path>/<to>/checkpoint-400000steps.pkl)

train_spks="ABA SKA YBAA BWC LXC NCC ASI RRBI SVBI TNI HJK HKK YDCK EBVS ERMS MBMPS HQTV PNV THV TLV"
eval_spks="NJS YKWK TXHC ZHAA"

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

train_set="train_nodev_$(echo "${train_spks}" | tr " " "_")" # name of training data directory
dev_set="dev_$(echo "${train_spks}" | tr " " "_")"           # name of development data directory
eval_set="eval_$(echo "${eval_spks}" | tr " " "_")"         # name of evaluation data directory

set -euo pipefail

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    train_data_dirs=""
    dev_data_dirs=""
    eval_data_dirs=""
    for spk in ${train_spks}; do
        local/data_prep.sh \
            --train_set "train_nodev_${spk}" \
            --dev_set "dev_${spk}" \
            --eval_set "eval_${spk}" \
            "${db_root}" "${spk}" data
        train_data_dirs+=" data/train_nodev_${spk}"
        dev_data_dirs+=" data/dev_${spk}"
    done
    for spk in ${eval_spks}; do
        local/data_prep.sh \
            --train_set "train_nodev_${spk}" \
            --dev_set "dev_${spk}" \
            --eval_set "eval_${spk}" \
            "${db_root}" "${spk}" data
        eval_data_dirs+=" data/eval_${spk}"
    done
    # shellcheck disable=SC2086
    utils/combine_data.sh "data/${train_set}" ${train_data_dirs}
    # shellcheck disable=SC2086
    utils/combine_data.sh "data/${dev_set}" ${dev_data_dirs}
    # shellcheck disable=SC2086
    utils/combine_data.sh "data/${eval_set}" ${eval_data_dirs}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Pre-calculation"
    echo "Statistics computation start. See the progress via data/${train_set}/compute_statistics.log."
    ${train_cmd} "data/${train_set}/compute_statistics.log" \
        compute_statistics.py \
            --config "${conf}" \
            --scp "data/${train_set}/wav.scp" \
            --dumpdir "data/${train_set}" \
            --verbose "${verbose}"
    echo "Successfully calculated statistics."
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Pre-calculating speaker embeddings"
    # pids=()
    # for name in "${train_set}" "${dev_set}"; do
    # (
    #     [ ! -e "data/${name}/spemb" ] && mkdir -p "data/${name}/spemb"
    #     utils/make_subset_data.sh "data/${name}" "${n_jobs}" "data/${name}/spemb"
    #     echo "Speaker embedding computation start. See the progress via data/${name}/spemb/extract_spemb.log."
    #     ${train_cmd} JOB=1:${n_jobs} "data/${name}/spemb/extract_spemb.JOB.log" \
    #         extract_spemb.py \
    #             --config "${conf}" \
    #             --scp "data/${name}/spemb/wav.JOB.scp" \
    #             --dumpdir "data/${name}/spemb/dump.JOB" \
    #             --verbose "${verbose}"
    #     echo "Successfully finished feature extraction of ${name} set."
    # ) &
    # pids+=($!)
    # done
    # i=0; for pid in "${pids[@]}"; do wait "${pid}" || ((++i)); done
    # [ "${i}" -gt 0 ] && echo "$0: ${i} background jobs are failed." && exit 1;
    # echo "Successfully finished speaker embedding extraction."

    for name in "${train_set}" "${dev_set}"; do
        spemb_scp=data/${name}/spemb.scp
        find "$(realpath data/${name}/spemb)" -name "*.h5" -follow | sort | while read -r filename; do
            id="$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
            echo "${id} ${filename}" >> "${spemb_scp}"
        done
    done
    echo "Successfully finished speaker embedding scp generation."
fi

if [ -z ${tag} ]; then
    expname=${upstream}_$(basename ${conf%.*})
else
    expname=${upstream}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "data/${train_set}/stats.h5" "${expdir}"
    if [ "${n_gpus}" -gt 1 ]; then
        echo "Not Implemented yet."
        # train="python -m seq2seq_vc.distributed.launch --nproc_per_node ${n_gpus} -c parallel-wavegan-train"
    else
        train="train.py"
    fi
    echo "Training start. See the progress via ${expdir}/train.log."
    ${cuda_cmd} --gpu "${n_gpus}" "${expdir}/train.log" \
        ${train} \
            --upstream ${upstream} \
            --config "${conf}" \
            --train-scp "data/${train_set}/wav.scp" \
            --train-spemb-scp "data/${train_set}/spemb.scp" \
            --dev-scp "data/${dev_set}/wav.scp" \
            --dev-spemb-scp "data/${dev_set}/spemb.scp" \
            --trg-stats "${expdir}/stats.h5" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 4 ] && [ "${stop_stage}" -ge 4 ]; then
    echo "Stage 4: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    for name in "eval"; do
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            decode.py \
                --scp "data/${name}/wav.scp" \
                --checkpoint "${checkpoint}" \
                --trg-stats "${expdir}/stats.h5" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    done
    echo "Successfully finished decoding."
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Objective Evaluation"

    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    for name in "eval"; do
        wavdir="${outdir}/${name}/wav"
        echo "Evaluation start. See the progress via ${outdir}/${name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/evaluation.log" \
            local/evaluate.py \
                --wavdir ${wavdir} \
                --data_root "${db_root}/vcc2020" \
                --f0_path "conf/f0.yaml"
        grep "Mean MCD" "${outdir}/${name}/evaluation.log"
    done
fi
