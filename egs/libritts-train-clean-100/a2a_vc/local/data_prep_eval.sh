#!/bin/bash

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

eval_set="eval"
shuffle=false
num_ref_samples=10

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
data_dir=$2
lists_dir=$3

[ ! -e "${data_dir}/${eval_set}" ] && mkdir -p "${data_dir}/${eval_set}"

# set filenames
eval_scp="${data_dir}/${eval_set}/wav.scp"

# check file existence
[ -e "${eval_scp}" ] && rm "${eval_scp}"

# make eval scp
python local/generate_eval_list.py \
    --rootdir "${db_root}" \
    --train_list "${lists_dir}/vcc2020_train_list.txt" \
    --eval_list "${lists_dir}/vcc2020_eval_list.txt" \
    --num_ref_samples ${num_ref_samples} \
    --out "${eval_scp}"

echo "Successfully prepared eval data."