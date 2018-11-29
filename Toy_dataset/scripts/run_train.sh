#!/usr/bin/env bash
python train.py \
--seq_len 20 \
--max_seq_len 20 \
--save save_flow \
--disc_pre_epoch 2 \
--gen_pre_epoch 10\
--adversarial_epoch 10
