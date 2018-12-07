#!/usr/bin/env bash
python train.py \
--seq_len 20 \
--max_seq_len 20 \
--save save_original \
--disc_pre_epoch 10 \
--gen_pre_epoch 150 \
--adversarial_epoch 2000

