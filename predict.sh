#!/bin/bash

python predict.py         \
    --model=MODEL.pth                                   \
    --input=INPUT                                       \
    --output=OUTPUT                                     \
    --viz=True                                          \
    --no-save=False                                     \
    --mask-threshold=0.5                                \
    --scale=0.5                                         \
    --bilinear=False                                    \
    --classes=2
