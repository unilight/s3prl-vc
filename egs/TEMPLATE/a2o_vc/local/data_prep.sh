#!/bin/bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

num_dev=100
num_eval=100
train_set="train_nodev"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
spk=$2
data_dir=$3

# check arguments
if [ $# != 3 ]; then
    echo "Usage: $0 <db_root> <spk> <data_dir>"
    echo "e.g.: $0 downloads TEF1 data"
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

[ ! -e "${data_dir}/all" ] && mkdir -p "${data_dir}/all"

# set filenames
scp="${data_dir}/all/wav.scp"
segments="${data_dir}/all/segments"

# check file existence
[ -e "${scp}" ] && rm "${scp}"
[ -e "${segments}" ] && rm "${segments}"

# make scp
find "$(realpath ${db_root}/wav)" -name "*.wav" -follow | sort | while read -r filename; do
    id="$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
    echo "${id} ${filename}" >> "${scp}"
done

# split
num_all=$(wc -l < "${scp}")
num_deveval=$((num_dev + num_eval))
num_train=$((num_all - num_deveval))
utils/split_data.sh \
    --num_first "${num_train}" \
    --num_second "${num_deveval}" \
    --shuffle "${shuffle}" \
    "${data_dir}/all" \
    "${data_dir}/${train_set}" \
    "${data_dir}/deveval"
utils/split_data.sh \
    --num_first "${num_dev}" \
    --num_second "${num_eval}" \
    --shuffle "${shuffle}" \
    "${data_dir}/deveval" \
    "${data_dir}/${dev_set}" \
    "${data_dir}/${eval_set}"

# remove tmp directories
rm -rf "${data_dir}/all"
rm -rf "${data_dir}/deveval"

echo "Successfully prepared data."