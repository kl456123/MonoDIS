#!/bin/bash


rm results/fv/*
rm results/data/*

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 2 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/nofocal \
    # --config /data/object/liangxiong/nofocal/faster_rcnn/kitti/kitti_config.json
# --rois_vis

# baseline 89.2
# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 26 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/faster_rcnn \
    # --config /data/object/liangxiong/faster_rcnn/faster_rcnn/kitti/kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 32 \
    # --net new_faster_rcnn \
    # --load_dir /data/object/liangxiong/exp_iouweights_hem

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 43 \
    # --net distance_faster_rcnn \
    # --load_dir /data/object/liangxiong/distance_center

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 53 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/faster_rcnn_detection

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 1 \
    # --net refine_faster_rcnn \
    # --load_dir /data/object/liangxiong/refine

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 25 \
    # --net new_faster_rcnn \
    # --load_dir /data/object/liangxiong/exp_iouweights_hem_great

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 2 \
    # --net rfcn \
    # --load_dir /data/object/liangxiong/rfcn

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net fpn \
    # --checkpoint 3257 \
    # --checkepoch 38 \
    # --load_dir /data/object/liangxiong/fpn

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 47 \
    # --net mono_3d_simpler \
    # --load_dir /data/object/liangxiong/mask_rcnn
    # --thresh 0.1
# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 40 \
    # --net mono_3d_final \
    # --load_dir /data/object/liangxiong/semantic_test2 \
    # --img_dir  /home/pengwu/mono3d/seq/frames \
    # --calib_file ./000001.txt

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 8 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/use_iou

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 25 \
    # --net semantic \
    # --load_dir /data/object/liangxiong/part05

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 24 \
    # --net semantic \
    # --load_dir /data/object/liangxiong/semantic_weights

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 38 \
    # --net three_iou \
    # --load_dir /data/object/liangxiong/three_iou_slow_ohem

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
# --net three_iou_org_ohem \
# --load_dir /data/object/liangxiong/three_iou_best_attention \
# --checkpoint 3257 \
# --checkepoch 24

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 50 \
    # --net double_iou \
    # --load_dir /data/object/liangxiong/double_iou \
    # --nms 0.7 \
    # --thresh 0.2

# no encoded
# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net mono_3d \
    # --load_dir /data/object/liangxiong/mono_3d_train \
    # --checkpoint 3257 \
    # --checkepoch 100

# encoded
# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net mono_3d_simpler \
    # --load_dir /data/object/liangxiong/mono_3d_angle_reg_3d_both \
    # --checkpoint 3257 \
    # --checkepoch 25

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net oft_4c \
    # --load_dir /data/object/liangxiong/oft_4c \
    # --checkpoint 6683 \
    # --checkepoch 7
    # --img_dir /data/2011_09_26/2011_09_26_drive_0009_sync/image_02/data/ \
    # --calib_file ./000000.txt

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net oft \
    # --load_dir /data/object/liangxiong/oft \
    # --checkpoint 6683 \
    # --checkepoch 155
# --calib_file /home/pengwu/Detection/000001.txt \
# --img_dir /data/liangxiong/yizhuang/2019_0107_140749/keyframes/

# --feat_vis True
# --img_path /home/pengwu/mono3d/seq/frames/1535193200792697000.jpg \
# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net ssd \
    # --load_dir /data/object/liangxiong/ssd \
    # --checkpoint 3257 \
    # --checkepoch 57

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net multibin_simpler \
    # --load_dir /data/object/liangxiong/mono_3d_angle_reg_2d \
    # --checkpoint 3257 \
    # --checkepoch 12

# 3d proj 2d detection
# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net mono_3d_better \
    # --load_dir /data/object/liangxiong/tmp \
    # --checkpoint 3257 \
    # --checkepoch 65

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net mono_3d_final \
    # --load_dir /data/object/liangxiong/mono_3d_final_both_noclip \
    # --checkpoint 3257 \
    # --checkepoch 70 \
    # --img_dir  /home/pengwu/mono3d/seq/frames \
    # --calib_file ./000001.txt

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net semantic \
    # --load_dir /data/object/liangxiong/semantic_coco \
    # --checkpoint 9255 \
    # --checkepoch 6

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net pr_net \
    # --load_dir /data/object/liangxiong/pr_net \
    # --checkpoint 3480 \
    # --checkepoch 13

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net semantic \
    # --load_dir /data/object/liangxiong/semantic_bdd \
    # --checkpoint 8000 \
    # --checkepoch 1 \
    # --data bdd

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net mono_3d_final_angle \
    # --load_dir /data/object/liangxiong/coco_pretrained_angle \
    # --checkpoint 89 \
    # --checkepoch 32 \
    # --img_dir /data/dm202_3w/left_img \
    # --calib_file ./000004.txt

CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    --net mono_3d_final_plus \
    --load_dir /data/object/liangxiong/test \
    --checkpoint 2968 \
    --checkepoch 5
    # --img_dir /data/liangxiong/training/image_02/0000/ \
    # --calib_file /data/liangxiong/training/calib/0000.txt
    # --img_dir /data/2011_09_26/2011_09_26_drive_0011_sync/image_02/data/ \
    # --calib_file ./000000.txt
    # --img_dir /data/dm202_3w/left_img \
    # --calib_file ./000004.txt
    # --img_dir /data/pengwu/seq/keyframes \
    # --calib_file ./000003.txt
    # --img_dir  /home/pengwu/mono3d/seq/frames \
    # --calib_file ./000002.txt
    

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 42 \
    # --net three_iou_org_ohem_second \
    # --load_dir /data/object/liangxiong/three_iou_best_second


# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 2 \
    # --net single_iou \
    # --load_dir /data/object/liangxiong/single_iou

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 60 \
    # --net iou_faster_rcnn \
    # --load_dir /data/object/liangxiong/iou

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net loss \
    # --load_dir /data/object/liangxiong/loss \
    # --checkpoint 3257 \
    # --checkepoch 7

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net LED \
    # --load_dir /data/object/liangxiong/LED_clip \
    # --checkpoint 3257 \
    # --checkepoch 10

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net three_iou_org_ohem \
    # --load_dir /data/object/liangxiong/delta \
    # --checkpoint 3257 \
    # --checkepoch 18

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 82 \
    # --net overlaps \
    # --load_dir /data/object/liangxiong/overlaps
