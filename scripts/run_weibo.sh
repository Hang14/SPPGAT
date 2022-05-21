#!/usr/bin/env bash
# -*- coding: utf-8 -*-

TASK=weibo
LR=2e-5
WEIGHT_DECAY=0.01
DROPOUT=0.2

CUDA_LAUNCH_BLOCKING=1 python run_tag.py -task ${TASK} \
-weight_decay ${WEIGHT_DECAY} \
-learning_rate ${LR} \
-dropout ${DROPOUT}