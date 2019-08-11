#!/usr/bin/env bash
python train.py \
--seq_len 1 \
--max_seq_len 20 \
--save Poster/save_flow_ct_350 \
--disc_pre_epoch 4 \
--gen_pre_epoch 120 \
--adversarial_epoch 350 \
--lbda 1.0
