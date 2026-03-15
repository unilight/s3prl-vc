#!/bin/bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false
num_ref_samples=10

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
data_dir=$2
lists_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root> <data_dir> <lists_dir>"
    echo "e.g.: $0 downloads data local/lists"
    echo ""
    echo "Options:"
    echo "    --dev_set: name of dev set (default=dev)."
    echo "    --eval_set: name of eval set (default=eval)."
    echo "    --shuffle: whether to perform shuffle in making dev / eval set (default=false)."
    exit 1
fi

set -euo pipefail

[ ! -e "${data_dir}/${train_set}" ] && mkdir -p "${data_dir}/${train_set}"
[ ! -e "${data_dir}/${dev_set}" ] && mkdir -p "${data_dir}/${dev_set}"

# set filenames
train_scp="${data_dir}/${train_set}/wav.scp"
dev_scp="${data_dir}/${dev_set}/wav.scp"

# # check file existence
# [ -e "${train_scp}" ] && rm "${train_scp}"
# [ -e "${dev_scp}" ] && rm "${dev_scp}"

# # make train scp
# while IFS= read -r number; do
#     spk=$(echo ${number} | cut -d "_" -f 1)
#     wavfile="${db_root}/vctk/wav48/${spk}/${number}.wav"
#     [ -e "${wavfile}" ] && echo "${number} ${wavfile}" >> "${train_scp}"
# done < "${lists_dir}/train_list.txt"

# echo "Successfully prepared train data scp."

# # make dev scp
# while IFS= read -r number; do
#     spk=$(echo ${number} | cut -d "_" -f 1)
#     wavfile="${db_root}/vctk/wav48/${spk}/${number}.wav"
#     [ -e "${wavfile}" ] && echo "${number} ${wavfile}" >> "${dev_scp}"
# done < "${lists_dir}/dev_list.txt"

# echo "Successfully prepared dev data scp."

###################################################

[ ! -e "${data_dir}/${eval_set}" ] && mkdir -p "${data_dir}/${eval_set}"

# set filenames
eval_scp="${data_dir}/${eval_set}/wav.scp"

# check file existence
[ -e "${eval_scp}" ] && rm "${eval_scp}"

# make eval scp
python local/generate_eval_list.py \
    --rootdir "${db_root}/vcc2020" \
    --train_list "${lists_dir}/vcc2020_train_list.txt" \
    --eval_list "${lists_dir}/vcc2020_eval_list.txt" \
    --num_ref_samples ${num_ref_samples} \
    --out "${eval_scp}"

echo "Successfully prepared eval data."
