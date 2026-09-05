#!/usr/bin/env bash
set -euo pipefail
cd /home/yingch/projects/FATST
output_root=/home/yingch/exp_outputs/r-2026-fatst/uvhf_prefix_tail_weather_20260905
mkdir -p "$output_root/logs"
gpu_index="$1"
shift
for horizon in "$@"; do
  CUDA_VISIBLE_DEVICES="$gpu_index" PYTHONHASHSEED=2021 /home/yingch/.conda/envs/moe/bin/python -u baselines/dlinear/train.py --dataset-root /home/yingch/dataset --dataset Weather --seq-len 608 --pred-len "$horizon" --batch-size 128 --epochs 50 --learning-rate 0.0001 --patience 8 --seed 2021 --init-mode pytorch_default --run-name TailAuditedDLinear --output-root "$output_root" --device cuda --skip-test > "$output_root/logs/h${horizon}.log" 2>&1
  date -Iseconds > "$output_root/logs/h${horizon}.done"
done
