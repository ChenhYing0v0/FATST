#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

model_name=TimeAlign
seq_len=720

# ETTh1-derived bootstrap settings. These are a source-informed starting point,
# not a tuned Exchange profile.
dropout=0.1
w_align=0.1
patch_num=24
local_margin=0.5

for pred_len in 96 192 336 720; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_${seq_len}_${pred_len} \
    --model "${model_name}" \
    --data custom \
    --features M \
    --freq d \
    --seq_len "${seq_len}" \
    --label_len 48 \
    --pred_len "${pred_len}" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --des Exp \
    --d_model 32 \
    --d_ff 32 \
    --train_epochs 10 \
    --learning_rate 0.0005 \
    --dropout "${dropout}" \
    --w_align "${w_align}" \
    --patch_num "${patch_num}" \
    --local_margin "${local_margin}" \
    --seed 2021 \
    --itr 1
done
