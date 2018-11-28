#!/bin/bash

export CONDA_HOME=/u/nlp/anaconda/ubuntu_16
export PATH=${CONDA_HOME}/bin:/usr/local/cuda/bin:$PATH:/u/nlp/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
export CUDNN_PATH=/usr/local/cuda/lib64/
source activate py36-amita
sh scripts/run_train.sh
