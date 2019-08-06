#!/bin/bash
SAMPLE=000001
DATA_DIR=/data/object/training

python utils/box_vis.py \
    --kitti ${DATA_DIR}/label_2/${SAMPLE}.txt \
    --img ${DATA_DIR}/image_2/${SAMPLE}.png \
    --calib ${DATA_DIR}/calib/${SAMPLE}.txt
