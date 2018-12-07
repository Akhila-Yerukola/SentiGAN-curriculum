#!/usr/bin/env bash
python train.py \
--seq_len 1 \
--max_seq_len 17 \
--save save_ct_17_l1 \
--disc_pre_epoch 4 \
--gen_pre_epoch 150 \
--adversarial_epoch 2000 \
--lbda 1

