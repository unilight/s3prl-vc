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

upstream=vq_wav2vec
conf=conf/taco2_ar.yaml

# dataset configuration
db_root=downloads
dumpdir=dump                # directory to dump full features
trgspk=TEF1
stats_ext=h5

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
                                       
# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

set -euo pipefail

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data and Pretrained Model Download"
    local/data_download.sh ${db_root}/vcc2020
    local/vocoder_download.sh ${db_root}
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    local/data_prep.sh \
        --train_set "train" \
        --dev_set "dev" \
        --eval_set "eval" \
        "${db_root}/vcc2020" "${trgspk}" "data" "local/lists"
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Pre-calculation"
    echo "Statistics computation start. See the progress via data/${trgspk}_train/compute_statistics.log."
    ${train_cmd} "data/${trgspk}_train/compute_statistics.log" \
        compute_statistics.py \
            --config "${conf}" \
            --scp "data/${trgspk}_train/wav.scp" \
            --dumpdir "data/${trgspk}_train" \
            --verbose "${verbose}"
    echo "Successfully calculated statistics."
fi

if [ -z ${tag} ]; then
    expname=${trgspk}_${upstream}_$(basename ${conf%.*})
else
    expname=${trgspk}_${upstream}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "data/${trgspk}_train/stats.h5" "${expdir}"
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
            --train-scp "data/${trgspk}_train/wav.scp" \
            --dev-scp "data/${trgspk}_dev/wav.scp" \
            --trg-stats "${expdir}/stats.h5" \
            --outdir "${expdir}" \
            --resume "${resume}" \
            --verbose "${verbose}"
    echo "Successfully finished training."
fi

if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    echo "Stage 3: Network decoding"
    # shellcheck disable=SC2012
    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    pids=()
    for name in "${srcspk}_dev" "${srcspk}_eval"; do
        [ ! -e "${outdir}/${name}" ] && mkdir -p "${outdir}/${name}"
        [ "${n_gpus}" -gt 1 ] && n_gpus=1
        echo "Decoding start. See the progress via ${outdir}/${name}/decode.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/decode.log" \
            vc_decode.py \
                --dumpdir "${dumpdir}/${name}/norm_${norm_name}" \
                --checkpoint "${checkpoint}" \
                --trg-stats "${expdir}/${trgspk}_stats.${stats_ext}" \
                --outdir "${outdir}/${name}" \
                --verbose "${verbose}"
        echo "Successfully finished decoding of ${name} set."
    done
    echo "Successfully finished decoding."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Objective Evaluation"

    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"
    for name in "${srcspk}_dev" "${srcspk}_eval"; do
        wavdir="${outdir}/${name}/wav"
        echo "Evaluation start. See the progress via ${outdir}/${name}/evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/evaluation.log" \
            local/evaluate.py \
                --wavdir ${wavdir} \
                --data_root "${db_root}/cmu_us_${trgspk}_arctic" \
                --trgspk ${trgspk} \
                --f0_path "conf/f0.yaml"
    done
fi