#!/bin/bash

export PYTHONUNBUFFERED=TRUE

BATCH_SIZE=16
N_EPOCHS=800
TRAIN_SCRIPT=diffusion_train_full.py

python3 -u ${TRAIN_SCRIPT} --batch_size ${BATCH_SIZE} --n_epochs ${N_EPOCHS}