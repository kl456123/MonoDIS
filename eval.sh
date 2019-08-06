#!/bin/bash

#PROG=/data/liangxiong/avod/scripts/offline_eval/kitti_native_eval/evaluate_object_3d_offline 
#PROG=/data/liangxiong/avod/scripts/offline_eval/kitti_native_eval/evaluate_object_3d_offline_05_iou
PROG=/home/zhixiang/kitti_eval/evaluate_object_3d_offline
$PROG /data/object/training/label_2/ results/
# ~/kitti_native_evaluation/evaluate_object /data/object/training/label_2/ results/
# /data/liangxiong/kitti_devkit/cpp/evaluate_object /data/object/training/label_2/ results/ 0.1
# /data/liangxiong/kitti_devkit/cpp/my_evaluate_object_3d_offline /data/object/training/label_2/ results/ 0.7
