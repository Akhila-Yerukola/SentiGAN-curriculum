#!/usr/bin/env bash
python train.py \
--seq_len 1 \
--max_seq_len 20 \
--save Poster_Results/save_start_4_120_450 \
--disc_pre_epoch 4 \
--gen_pre_epoch 120 \
--adversarial_epoch 450
