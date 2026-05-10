#!/bin/bash

export PYTHONUNBUFFERED=TRUE

python3 -u mla_alignment.py \
    --learning_rate 5e-4 \
    --batch_size 16 \
    --appro_alpha 1.0 \
    --ot_weight 0.0 \
    --kl_weight 1.0 \
    --epoches 500