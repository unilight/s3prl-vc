#!/bin/bash

# Copyright 2022 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

# shellcheck disable=SC1091
. ./path.sh || exit 1;

train_set="train"
dev_set="dev"
eval_set="eval"
shuffle=false

# shellcheck disable=SC1091
. utils/parse_options.sh || exit 1;

db_root=$1
spk=$2
emo=$3
data_dir=$4

# check arguments
if [ $# != 4 ]; then
    echo "Usage: $0 <db_root> <spk> <emotion> <data_dir>"
    echo "e.g.: $0 downloads/cms_us_slt_arctic slt data"
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

[ ! -e "${data_dir}/${spk}_${emo}_train" ] && mkdir -p "${data_dir}/${spk}_${emo}_train"
[ ! -e "${data_dir}/${spk}_${emo}_dev" ] && mkdir -p "${data_dir}/${spk}_${emo}_dev"
[ ! -e "${data_dir}/${spk}_${emo}_eval" ] && mkdir -p "${data_dir}/${spk}_${emo}_eval"

# set filenames
train_scp="${data_dir}/${spk}_${emo}_train/wav.scp"
dev_scp="${data_dir}/${spk}_${emo}_dev/wav.scp"
eval_scp="${data_dir}/${spk}_${emo}_eval/wav.scp"

# check file existence
[ -e "${train_scp}" ] && rm "${train_scp}"
[ -e "${dev_scp}" ] && rm "${dev_scp}"
[ -e "${eval_scp}" ] && rm "${eval_scp}"

# make scp
find "$(realpath ${db_root}/train)" -name "*.wav" -follow | sort | while read -r filename; do
    id="$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
    original_number=$(echo ${id} | cut -d "_" -f 2)
    renumber=$((10#${original_number} % 350))
    echo "${renumber} ${filename}" >> "${train_scp}"
done
find "$(realpath ${db_root}/test)" -name "*.wav" -follow | sort | while read -r filename; do
    id="$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
    original_number=$(echo ${id} | cut -d "_" -f 2)
    renumber=$((10#${original_number} % 350))
    echo "${renumber} ${filename}" >> "${dev_scp}"
done
find "$(realpath ${db_root}/evaluation)" -name "*.wav" -follow | sort | while read -r filename; do
    id="$(basename "${filename}" | sed -e "s/\.[^\.]*$//g")"
    original_number=$(echo ${id} | cut -d "_" -f 2)
    renumber=$((10#${original_number} % 350))
    echo "${renumber} ${filename}" >> "${eval_scp}"
done

echo "Successfully prepared ${spk} ${emo} data."
