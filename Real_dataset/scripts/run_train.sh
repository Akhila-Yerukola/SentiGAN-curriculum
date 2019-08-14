#!/usr/bin/env bash
python train.py \
--seq_len 17 \
--max_seq_len 17 \
--save flow_len17_dis10_gen150_adv350 \
--disc_pre_epoch 10 \
--gen_pre_epoch 150 \
--adversarial_epoch 350 \
--lbda 1.0

