#!/bin/bash

SAMPLE_IDX=000000
CHECKEPOCH=42
NMS=0.5
THRESH=0.1

# python test_net.py \
    # --cuda \
    # --net double_iou_second \
    # --checkpoint 3257 \
    # --nms ${NMS} \
    # --thresh ${THRESH} \
    # --checkepoch ${CHECKEPOCH} \
    # --load_dir /data/object/liangxiong/double_iou_second \
    # --img_path /data/object/training/image_2/${SAMPLE_IDX}.png \
    # --feat_vis True

# python utils/visualize.py \
    # --kitti results/data/${SAMPLE_IDX}.txt \
    # --img /data/object/training/image_2/${SAMPLE_IDX}.png
# --thresh ${THRESH} \
DIR='/home/pengwu/mono3d/seq/frames'
SAMPLE_IDX=1535193187493031000
# CUDA_VISIBLE_DEVICES=0 python test_net.py \
        # --cuda \
        # --net oft \
        # --checkpoint 3257 \
        # --nms ${NMS} \
        # --checkepoch ${CHECKEPOCH} \
        # --load_dir /data/object/liangxiong/tmp/oft \
        # --img_path ${DIR}/${SAMPLE_IDX}.jpg \
        # --feat_vis False
python utils/visualize.py \
        --kitti /data/object/liangxiong/second/eval_results/step_30950/000008.txt \
        --img /data/object/training/image_2/000008.png
# for file in ${DIR}/*
# do
    # TMP=${file##*/}
    # SAMPLE_IDX=${TMP:0:6}
    # echo ${SAMPLE_IDX}
    # python test_net.py \
        # --cuda \
        # --net mono_3d \
        # --checkpoint 3257 \
        # --nms ${NMS} \
        # --thresh ${THRESH} \
        # --checkepoch ${CHECKEPOCH} \
        # --load_dir /data/object/liangxiong/tmp/oft \
        # --img_path ${DIR}/${SAMPLE_IDX}.jpg \
        # --feat_vis False
    # python utils/visualize.py \
        # --kitti results/data/${SAMPLE_IDX}.txt \
        # --img /home/pengwu/mono3d/kitti/0006/${SAMPLE_IDX}.png
# done



