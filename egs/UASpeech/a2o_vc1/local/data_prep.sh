#!/bin/bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
spk=$2
data_dir=$3
lists_dir=$4

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <db_root> <spk> <data_dir> <lists_dir>"
    echo "e.g.: $0 downloads/vcc2020 TEF1 data local/lists"
    echo ""
    echo "Options:"
    echo "    --num_dev: number of development uttreances (default=100)."
    echo "    --num_eval: number of evaluation uttreances (default=100)."
    echo "    --train_set: name of train set (default=train_nodev)."
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    echo "    --shuffle: whether to perform shuffle in making dev / eval set (default=false)."
    exit 1
fi

set -euo pipefail

dyspks=(
    "M04" "M05" "M08" "M10" "M11" "M12"
)
ctrlspks=(
    "CF02" "CF04" "CM06" "CM08"
)

# check speaker
# if ! echo "${trgspks[*]}" | grep -q "${spk}"; then
#     echo "Specified speaker ${spk} is not available."
#     echo "Available speakers: ${trgspks[*]}"
#     exit 1
# fi

[ ! -e "${data_dir}/${spk}_${train_set}" ] && mkdir -p "${data_dir}/${spk}_${train_set}"
[ ! -e "${data_dir}/${spk}_${dev_set}" ] && mkdir -p "${data_dir}/${spk}_${dev_set}"

# set filenames
train_scp="${data_dir}/${spk}_${train_set}/wav.scp"
dev_scp="${data_dir}/${spk}_${dev_set}/wav.scp"

# check file existence
[ -e "${train_scp}" ] && rm "${train_scp}"
[ -e "${dev_scp}" ] && rm "${dev_scp}"

# make train scp
while IFS= read -r number; do
    wavfile="${db_root}/${spk}/${spk}_${number}.wav"
    [ -e "${wavfile}" ] && echo "${number} ${wavfile}" >> "${train_scp}"
done < "${lists_dir}/train_list.txt"

echo "Successfully prepared train data scp."

# make dev scp
while IFS= read -r number; do
    wavfile="${db_root}/${spk}/${spk}_${number}.wav"
    [ -e "${wavfile}" ] && echo "${number} ${wavfile}" >> "${dev_scp}"
done < "${lists_dir}/dev_list.txt"

echo "Successfully prepared dev data scp."

###################################################

[ ! -e "${data_dir}/${eval_set}" ] && mkdir -p "${data_dir}/${eval_set}"

# determine direction
is_ctrl=$(echo ${spk} | head -c 1)
if [ "${is_ctrl}" = "C" ]; then
    direction="N2D"
else
    direction="D2N"
fi

# set filenames
eval_scp="${data_dir}/${eval_set}/${direction}_wav.scp"

# check file existence
if [ ! -e "${eval_scp}" ]; then
    # make eval scp
    while IFS= read -r number; do
        # loop through speakers based on direction
        # if N2D, then target speaker is ctrl spk, then source is dysarthric
        # if D2N, then target speaker is dy spk, then source is ctrl
        if [ "${direction}" = "N2D" ]; then
            for srcspk in "${dyspks[@]}"; do
                wavfile="${db_root}/${srcspk}/${srcspk}_${number}.wav"
                [ -e "${wavfile}" ] && echo "${srcspk}_${number} ${wavfile}" >> "${eval_scp}"
            done 
        else
            for srcspk in "${ctrlspks[@]}"; do
                wavfile="${db_root}/${srcspk}/${srcspk}_${number}.wav"
                [ -e "${wavfile}" ] && echo "${srcspk}_${number} ${wavfile}" >> "${eval_scp}"
            done 
        fi
    done < "${lists_dir}/test_list.txt"
fi

echo "Successfully prepared eval data."
