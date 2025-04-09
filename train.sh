#!/bin/bash

DATE_TIME=$(date +%Y%m%d%H%M%S)

python train.py         \
    --epochs=5                                          \
    --batch-size=4                                      \
    --learning-rate=1e-5                                \
    --load=checkpoints\checkpoint_epoch2200.pth         \
    --scale=0.1                                         \
    --validation=10                                     \
    --amp=False                                         \
    --bilinear=False                                    \
    --classes=2                                         \
    --channels=3                                        \
    --dir-img=data/imgs                                 \
    --dir-mask=data/masks                               \
    --dir-output=checkpoints | tee ${DATE_TIME}.log &
