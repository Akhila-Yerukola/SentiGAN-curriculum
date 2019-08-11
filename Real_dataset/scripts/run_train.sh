#!/usr/bin/env bash
python train.py \
--seq_len 17 \
--max_seq_len 17 \
--save save_original_17_flow_true \
--disc_pre_epoch 10 \
--gen_pre_epoch 150 \
--adversarial_epoch 2000 \
--lbda 1.0

