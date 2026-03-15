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
conf=conf/taco2_ar_f0.yaml

# dataset configuration
db_root=/home/z44476r/data/Corpora/esd
spk=0013
trgemo=Neutral

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

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    # local/data_prep.sh "${db_root}/${spk}/${trgemo}" "${spk}" "${trgemo}" data

    for _spk in 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020; do
        for _emo in Neutral Angry Happy Sad Surprise; do
            local/data_prep.sh "${db_root}/${_spk}/${_emo}" "${_spk}" "${_emo}" data
            ${train_cmd} "data/${_spk}_${_emo}_train/create_histogram.log" \
                create_histogram.py \
                    --scp "data/${_spk}_${_emo}_train/wav.scp" \
                    --figure_dir "data/${_spk}_${_emo}_train"
        done
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Pre-calculation"
    echo "Statistics computation start. See the progress via data/${spk}_${trgemo}_train/compute_statistics.log."
    ${train_cmd} "data/${spk}_${trgemo}_train/compute_statistics.log" \
        compute_statistics.py \
            --config "${conf}" \
            --scp "data/${spk}_${trgemo}_train/wav.scp" \
            --dumpdir "data/${spk}_${trgemo}_train" \
            --verbose "${verbose}" \
            --f0 --f0_path "conf/f0.yaml" --spk ${spk}_${trgemo}
    echo "Successfully calculated statistics."
fi

if [ -z ${tag} ]; then
    expname=${spk}_${trgemo}_${upstream}_$(basename ${conf%.*})
else
    expname=${spk}_${trgemo}_${upstream}_${tag}
fi
expdir=exp/${expname}
if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "Stage 2: Network training"
    [ ! -e "${expdir}" ] && mkdir -p "${expdir}"
    cp "data/${spk}_${trgemo}_train/stats.h5" "${expdir}"
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
            --train-scp "data/${spk}_${trgemo}_train/wav.scp" \
            --dev-scp "data/${spk}_${trgemo}_dev/wav.scp" \
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
    for _spk in 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020; do
        for _emo in Neutral Angry Happy Sad Surprise; do
            name=${_spk}_${_emo}_eval
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
    done
    echo "Successfully finished decoding."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Objective Evaluation"

    [ -z "${checkpoint}" ] && checkpoint="$(ls -dt "${expdir}"/*.pkl | head -1 || true)"
    outdir="${expdir}/results/$(basename "${checkpoint}" .pkl)"

    # EPSVC
    for _spk in 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020; do
        name=${_spk}_${trgemo}_eval
        wavdir="${outdir}/${name}/wav"

        echo "EPSVC evaluation start. See the progress via ${outdir}/${name}/epsvc_evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/epsvc_evaluation.log" \
            local/evaluate.py \
                --wavdir ${wavdir} \
                --data_root "${db_root}" \
                --srcemo ${trgemo} \
                --trgspk ${spk} \
                --trgemo ${trgemo} \
                --f0_path "conf/f0.yaml" \
                --scenario "epsvc"
        grep "Mean MCD" "${outdir}/${name}/epsvc_evaluation.log"
    done
    python local/parse_results.py --dir ${outdir} --trgspk ${spk} --trgemo ${trgemo} --scenario epsvc

    # SPEVC
    for _emo in Neutral Angry Happy Sad Surprise; do
        name=${spk}_${_emo}_eval
        wavdir="${outdir}/${name}/wav"

        echo "SPEVC evaluation start. See the progress via ${outdir}/${name}/spevc_evaluation.log."
        ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/spevc_evaluation.log" \
            local/evaluate.py \
                --wavdir ${wavdir} \
                --data_root "${db_root}" \
                --srcemo ${_emo} \
                --trgspk ${spk} \
                --trgemo ${trgemo} \
                --f0_path "conf/f0.yaml" \
                --scenario "spevc"
        grep "Mean MCD" "${outdir}/${name}/spevc_evaluation.log"
    done
    python local/parse_results.py --dir ${outdir} --trgspk ${spk} --trgemo ${trgemo} --scenario spevc
    

    
    # for _spk in 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020; do
    #     for _emo in Neutral Angry Happy Sad Surprise; do
    #         name=${_spk}_${_emo}_eval
    #         wavdir="${outdir}/${name}/wav"

    #         echo "SPEVC evaluation start. See the progress via ${outdir}/${name}/spevc_evaluation.log."
    #         ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/spevc_evaluation.log" \
    #             local/evaluate.py \
    #                 --wavdir ${wavdir} \
    #                 --data_root "${db_root}" \
    #                 --srcemo ${_emo} \
    #                 --trgspk ${spk} \
    #                 --trgemo ${trgemo} \
    #                 --f0_path "conf/f0.yaml" \
    #                 --scenario "spevc"
    #         grep "Mean MCD" "${outdir}/${name}/spevc_evaluation.log"

    #         echo "EPSVC evaluation start. See the progress via ${outdir}/${name}/epsvc_evaluation.log."
    #         ${cuda_cmd} --gpu "${n_gpus}" "${outdir}/${name}/epsvc_evaluation.log" \
    #             local/evaluate.py \
    #                 --wavdir ${wavdir} \
    #                 --data_root "${db_root}" \
    #                 --srcemo ${_emo} \
    #                 --trgspk ${spk} \
    #                 --trgemo ${trgemo} \
    #                 --f0_path "conf/f0.yaml" \
    #                 --scenario "epsvc"
    #         grep "Mean MCD" "${outdir}/${name}/epsvc_evaluation.log"
    #     done
    # done
fi