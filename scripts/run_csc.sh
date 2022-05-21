#!/usr/bin/env bash
# -*- coding: utf-8 -*-

TASK=chn_senti_corp
WEIGHT_DECAY=0.01
LR=2e-5
DROPOUT=0.2

CUDA_LAUNCH_BLOCKING=1 python run_cls.py -task ${TASK} \
-weight_decay ${WEIGHT_DECAY} \
-learning_rate ${LR} \
-dropout ${DROPOUT}