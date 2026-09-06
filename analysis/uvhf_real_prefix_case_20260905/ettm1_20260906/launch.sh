#!/usr/bin/env bash
set -euo pipefail
cd /home/yingch/projects/FATST
output_root=/home/yingch/exp_outputs/r-2026-fatst/uvhf_prefix_ettm1_20260906
mkdir -p "$output_root/logs"
gpu_index="$1"
shift
for horizon in "$@"; do
  CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONHASHSEED=2021 /home/yingch/.conda/envs/moe/bin/python -u analysis/uvhf_real_prefix_case_20260905/tail_audited/train_timemixer.py --dataset ETTm1 --seq-len 96 --batch-size 16 --horizon "$horizon" --learning-rate 0.01 --output "$output_root" > "$output_root/logs/h${horizon}.log" 2>&1
  date -Iseconds > "$output_root/logs/h${horizon}.done"
done
