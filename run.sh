#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net two_rpn \
    # --out_path /data/object/liangxiong/two_rpn \
    # --config configs/two_rpn_config.json
# --r True \
    # --checkpoint 3257 \
    # --checkepoch 1


# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net distance_faster_rcnn \
    # --out_path /data/object/liangxiong/distance \
    # --config configs/kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net distance_faster_rcnn \
    # --out_path /data/object/liangxiong/distance_center \
    # --config configs/distance_center_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net rfcn \
    # --out_path /data/object/liangxiong/rfcn \
    # --config configs/rfcn_kitti_config.json
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net double_iou_second \
    # --out_path /data/object/liangxiong/double_iou_second \
    # --config configs/double_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net double_iou_second \
    # --out_path /data/object/liangxiong/double_iou_second \
    # --config configs/double_iou_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 13 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net fpn \
    # --out_path /data/object/liangxiong/fpn \
    # --config configs/fpn_config.json

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net double_iou_second \
    # --out_path /data/object/liangxiong/double_iou_third \
    # --config configs/double_iou_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 1 \
    # --r True
# --model /data/object/liangxiong/double_iou_second/double_iou_second/kitti/faster_rcnn_45_3257.pth

# --model /data/object/liangxiong/double_iou_new/double_iou/kitti/faster_rcnn_100_3257.pth
# --checkpoint 3257 \
    # --checkepoch 42 \
    # --r True
# --model /data/object/liangxiong/double_iou_new/double_iou/kitti/faster_rcnn_100_3257.pth
# --checkpoint 3257 \
    # --checkepoch 62 \
    # --r True
# --model /data/object/liangxiong/double_iou/double_iou/kitti/faster_rcnn_53_3257.pth \
    # --lr 1e-5

# no encoded
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net mono_3d \
    # --out_path /data/object/liangxiong/mono_3d_train \
    # --config configs/refine_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 100

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net mono_3d \
    # --out_path /data/object/liangxiong/tmp \
    # --config configs/refine_kitti_config.json \
    # --model /data/object/liangxiong/semantic/multibin/kitti/faster_rcnn_50_3257.pth
    # --lr 1e-2

# 2d box
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net mono_3d_better \
    # --out_path /data/object/liangxiong/mono_3d_better \
    # --config configs/mono_3d_config.json
    # --checkpoint 3257 \
    # --checkepoch 50 \
    # --r True
    # --model /data/object/liangxiong/semantic/multibin/kitti/faster_rcnn_50_3257.pth
    # --model /data/object/liangxiong/semantic_3d/multibin/kitti/faster_rcnn_40_3257.pth
# --model /data/object/liangxiong/faster_rcnn/mono_3d/kitti/faster_rcnn_53_3257.pth

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net mono_3d_simpler \
    # --out_path /data/object/liangxiong/mask_rcnn \
    # --config configs/refine_kitti_config.json \
    # --model /data/object/liangxiong/semantic_test/semantic/kitti/faster_rcnn_50_3257.pth

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net pr_net \
    # --out_path /data/object/liangxiong/pr_net \
    # --config configs/pr_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net mono_3d_final_plus \
    # --out_path /data/object/liangxiong/mono_3d_final_plus \
    # --config configs/mono_3d_config.json

CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    --net mono_3d_final_plus \
    --out_path /data/object/liangxiong/test \
    --config configs/coco_mono_3d_config.json \
    --model /data/object/liangxiong/pretrained_models/mono_3d_final_plus/kitti/faster_rcnn_30_1518.pth
    # --checkpoint 2388 \
    # --checkepoch 15 \
    # --r True
    # --model /data/object/liangxiong/pretrained_models/mono_3d_final_plus/kitti/faster_rcnn_10_2678.pth
    # --checkpoint 1518 \
    # --checkepoch 30 \
    # --r True
    # --model /data/object/liangxiong/semantic_coco/semantic/kitti/faster_rcnn_9_70008.pth

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net mono_3d_final_angle \
    # --out_path /data/object/liangxiong/coco_pretrained_angle \
    # --config configs/coco_mono_3d_config.json \
    # --model /data/object/liangxiong/semantic_coco/semantic/kitti/faster_rcnn_9_70008.pth

# --checkpoint 162 \
# --r True \
# --checkepoch 12
    # --model /data/object/liangxiong/semantic_test2/mono_3d_final/kitti/faster_rcnn_40_3257.pth

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/semantic_coco \
    # --config configs/refine_kitti_config.json
    # --r True \
    # --checkepoch 1 \
    # --checkpoint 6000

# --model /data/object/liangxiong/semantic_test2/mono_3d_final/kitti/faster_rcnn_40_3257.pth

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net mono_3d_final \
    # --out_path /data/object/liangxiong/mono_3d_final_both \
    # --config configs/mono_3d_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/semantic_test2 \
    # --config configs/mono_3d_config.json
# --model /data/object/liangxiong/semantic/multibin/kitti/faster_rcnn_50_3257.pth

# --model /data/object/liangxiong/mono_3d_angle_reg_2d/multibin_simpler/kitti/faster_rcnn_10_3257.pth
    # --checkpoint 3257 \
    # --checkepoch 10 \
    # --r True

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net ssd \
    # --out_path /data/object/liangxiong/ssd \
    # --config configs/ssd_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net multibin_new \
    # --out_path /data/object/liangxiong/mono_3d_angle_reg_2d \
    # --config configs/refine_kitti_config.json \
    # --model /data/object/liangxiong/semantic/multibin/kitti/faster_rcnn_50_3257.pth
# --checkepoch 9 \
# --r True \
# --checkpoint 3257
# --checkepoch 3 \
# --r True \
# --checkpoint 3257
    # --model /data/object/liangxiong/faster_rcnn/mono_3d/kitti/faster_rcnn_53_3257.pth

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net multibin \
    # --out_path /data/object/liangxiong/mono_3d_multibin \
    # --config configs/refine_kitti_config.json \
    # --model /data/object/liangxiong/faster_rcnn_3d/mono_3d/kitti/faster_rcnn_60_3257.pth

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net multibin \
    # --out_path /data/object/liangxiong/mono_3d_angle_reg_2d \
    # --config configs/refine_kitti_config.json \
    # --r True \
    # --checkpoint 3257 \
    # --checkepoch 36
# --model /data/object/liangxiong/semantic/multibin/kitti/faster_rcnn_27_3257.pth

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net multibin_new \
    # --out_path /data/object/liangxiong/mono_3d_angle_reg_2d_both \
    # --config configs/refine_kitti_config.json \
    # --model /data/object/liangxiong/semantic/multibin/kitti/faster_rcnn_50_3257.pth

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net oft \
    # --out_path /data/object/liangxiong/oft \
    # --config configs/oft_config.json \
    # --model /data/object/liangxiong/tmp/oft/oft/kitti/faster_rcnn_27_6683.pth
    # --lr 1e-7 \
    # --checkpoint 3257 \
    # --checkepoch 49 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/demo \
    # --config configs/refine_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net oft_4c \
    # --out_path /data/object/liangxiong/oft_4c \
    # --config configs/oft_config.json
    # --model /data/object/liangxiong/oft/oft/kitti/faster_rcnn_100_3257.pth \
    # --lr 1e-7
    # --checkpoint 3257 \
    # --checkepoch 100 \
    # --r True
    # --model /data/object/liangxiong/semantic/multibin/kitti/faster_rcnn_50_3257.pth

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net mono_3d \
    # --out_path /data/object/liangxiong/mono_3d_angle_reg_3d \
    # --config configs/refine_kitti_config.json \
    # --model /data/object/liangxiong/semantic/multibin/kitti/faster_rcnn_50_3257.pth
# --checkpoint 3257 \
# --checkepoch 13 \
# --r True


# 3d proj
# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net mono_3d \
    # --out_path /data/object/liangxiong/mono_3d_angle_reg \
    # --config configs/refine_kitti_config.json
# --checkepoch 6 \
# --checkpoint 3257 \
# --r True

# encode
# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net mono_3d \
    # --out_path /data/object/liangxiong/mono_3d_train_encode_better \
    # --config configs/refine_kitti_config.json
    # --model /data/object/liangxiong/double_iou/double_iou/kitti/faster_rcnn_53_3257.pth \
    # --lr 1e-5
# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net three_iou_org_ohem \
    # --out_path /data/object/liangxiong/three_iou_org_ohem \
    # --config configs/org_three_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net three_iou_org_ohem_second \
    # --out_path /data/object/liangxiong/three_iou_best_second \
    # --config configs/org_three_iou_kitti_config.json \
    # --model /data/object/liangxiong/double_iou_new/double_iou/kitti/faster_rcnn_100_3257.pth
# --checkpoint 3257 \
    # --checkepoch 3 \
    # --r True
# --model /data/object/liangxiong/three_iou_best/three_iou_org_ohem/kitti/faster_rcnn_84_3257.pth
# --model /data/object/liangxiong/three_iou_/three_iou/kitti/faster_rcnn_39_3257.pth
# --checkpoint 3257 \
    # --checkepoch 7 \
    # --r True
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net three_iou_org \
    # --out_path /data/object/liangxiong/three_iou_org \
    # --config configs/org_three_iou_kitti_config.json \
    # --checkepoch 23 \
    # --checkpoint 3257 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net three_iou_org_ohem \
    # --out_path /data/object/liangxiong/three_iou_attention \
    # --config configs/org_three_iou_kitti_config.json
# --checkepoch 44 \
    # --checkpoint 3257 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net cascade \
    # --out_path /data/object/liangxiong/cascade \
    # --config configs/cascade_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net three_iou_org \
    # --out_path /data/object/liangxiong/three_iou_org_ohem \
    # --config configs/org_three_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net double_iou_slow \
    # --out_path /data/object/liangxiong/double_iou_slow_ohem \
    # --config configs/double_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net double_iou_slow \
    # --out_path /data/object/liangxiong/double_iou_slow \
    # --config configs/double_iou_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 29 \
    # --r True

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net double_iou_slow \
    # --out_path /data/object/liangxiong/double_iou_slow_ohem_better \
    # --config configs/double_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net double_iou_slow \
    # --out_path /data/object/liangxiong/double_iou_slow_01 \
    # --config configs/double_iou_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 17 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net single_iou \
    # --out_path /data/object/liangxiong/single_iou \
    # --config configs/refine_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 7 \
    # --r True

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/part07 \
    # --config configs/refine_kitti_config.json \
    # --model /data/object/liangxiong/semantic/semantic/kitti/faster_rcnn_100_3257.pth
# --model /data/object/liangxiong/part05/semantic/kitti/faster_rcnn_24_3257.pth
# --checkpoint 3257 \
    # --checkepoch 25 \
    # --r True

# --model /data/object/liangxiong/faster_rcnn/faster_rcnn/kitti/faster_rcnn_53_3257.pth

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/semantic_self \
    # --config configs/refine_kitti_config.json
# --checkpoint 3257 \
    # --checkepoch 130 \
    # --r True \
    # --lr 0.5

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net new_semantic \
    # --out_path /data/object/liangxiong/semantic_new \
    # --config configs/refine_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 13 \
    # --r True

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/semantic_anchors \
    # --config configs/refine_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 47 \
    # --r True

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net new_faster_rcnn \
    # --out_path /data/object/liangxiong/exp_iouweights_hem_great \
    # --config configs/kitti_config.json
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net loss \
    # --out_path /data/object/liangxiong/loss \
    # --config configs/refine_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net LED \
    # --out_path /data/object/liangxiong/LED_clip \
    # --config configs/LED_kitti_config.json \
    # --model /data/object/liangxiong/faster_rcnn/faster_rcnn/kitti/faster_rcnn_53_3257.pth \
    # --lr 1e-3

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net iou_faster_rcnn \
    # --out_path /data/object/liangxiong/iou_exp \
    # --config configs/iou_kitti_config.json
# --model /data/object/liangxiong/semantic/semantic/kitti/faster_rcnn_24_3257.pth
# --checkpoint 3257 \
    # --checkepoch 5 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net overlaps \
    # --out_path /data/object/liangxiong/overlaps \
    # --config configs/overlaps_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 10 \
    # --r True
