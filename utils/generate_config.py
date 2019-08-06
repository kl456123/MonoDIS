# -*- coding: utf-8 -*-
"""
The script is used for generating json configuration file.
"""

# some shared parameters configurations
coder_config = {
    "bbox_normalize_targets_precomputed": True,
    "bbox_inside_weights": [1.0, 1.0, 1.0, 1.0],
    "bbox_normalize_stds": [0.1, 0.1, 0.2, 0.2],
    "bbox_normalize_means": [0.0, 0.0, 0.0, 0.0]
}
classes = ['bg', 'Car']
num_classes = len(classes)

class_agnostic = True

normal_mean = [0.485, 0.456, 0.406]
normal_van = [0.229, 0.224, 0.225]
pretained_model = ""
img_channels = 3
data_root_path = "/data/object/training"
use_focal_loss = True

target_assigner_config = {"coder_config": {}, "matcher_config": {}}
sampler_config = {}
anchor_generator_config = {
    "base_anchor_size": 16,
    "scales": [2, 3, 4],
    "aspect_ratios": [0.5, 1, 2],
    "anchor_stride": [16, 16],
    "anchor_offset": [0, 0]
}
matcher_config = {}

rpn_config = {
    "rpn_batch_size": 1024,
    "anchor_generator_config": anchor_generator_config,
    "sampler_config": {
        "fg_fraction": 0.25
    },
    "target_assigner_config": {
        "coder_config": coder_config,
        "matcher_config": {
            "type": "bipartitle",
            "thresh": 0.5
        }
    },
    "num_bbox_samples": 500,
    "num_cls_samples": 2000,
    "din": 1024,
    "pre_nms_topN": 12000,
    "post_nms_topN": 1024,
    "nms_thresh": 0.7,
    "min_size": 16,
    "use_score": False
}
model_config = {
    "rpn_config": rpn_config,
    "subsample_twice": False,
    "use_focal_loss": use_focal_loss,
    "num_classes": num_classes,
    "classes": classes,
    "class_agnostic": class_agnostic,
    "pooling_size": 7,
    "pooling_mode": "align",
    "crop_resize_with_max_pool": False,
    "truncated": False,
    "target_assigner_config": {
        "coder_config": coder_config,
        "matcher_config": {
            "type": "argmax",
            "thresh": 0.5
        }
    },
    "sampler_config": {
        "fg_fraction": 0.25
    },
    "feature_extractor_config": {
        "pretained_model": pretained_model,
        "class_agnostic": class_agnostic,
        "img_channels": img_channels,
        "classes": classes,
        "pretrained": True
    },
    "rcnn_batch_size": 512
}

data_config = {
    "name": "kitti",
    "dataset_config": {
        "root_path": data_root_path,
        "dataset_file": "train.txt",
        "cache_bev": False
    },
    "transform_config": {
        "normal_mean": normal_mean,
        "normal_van": normal_van,
        "resize_range": [0.2, 0.4],
        "random_brightness": 10,
        "crop_size": [284, 1300],
        "random_blur": 0
    },
    "dataloader_config": {
        "shuffle": True,
        "batch_size": 1,
        "num_workers": 1
    }
}
eval_data_config = {
    "name": "kitti",
    "dataset_config": {
        "root_path": data_root_path,
        "dataset_file": "val.txt",
    },
    "transform_config": {
        "normal_mean": normal_mean,
        "normal_van": normal_van,
    },
    "dataloader_config": {
        "shuffle": False,
        "batch_size": 1,
        "num_workers": 1
    }
}

train_config = {
    "rng_seed": 3,
    "device_ids": [0],
    "disp_interval": 100,
    "max_epochs": 100,
    "checkpoint_interval": 10000,
    "mGPUs": True,
    "clip_gradient": 10,
    "start_epoch": 1,
    "scheduler_config": {
        "type": "step",
        "lr_decay_gamma": 0.1,
        "lr_decay_step": 20,
        "last_epoch": -1
    },
    "optimizer_config": {
        "type": "sgd",
        "momentum": 0.9,
        "lr": 0.001
    }
}

eval_config = {
    "rng_seed": 3,
    "max_per_image": 100,
    "bbox_reg": True,
    "bbox_normalize_targets_precomputed":
    coder_config['bbox_normalize_targets_precomputed'],
    "bbox_normalize_stds": coder_config['bbox_normalize_stds'],
    "bbox_normalize_means": coder_config['bbox_normalize_means'],
    "batch_size": 1,
    "class_agnostic": class_agnostic,
    "thresh": 0.5,
    "nms": 0.3,
    "classes": classes,
    "eval_out": "./results/data"
}

config = {
    "model_config": model_config,
    "data_config": data_config,
    "eval_data_config": eval_data_config,
    "eval_config": eval_config,
    "train_config": train_config
}


def generate_config(json_file, config):
    import json
    with open(json_file, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    json_file = './configs/kitti_config.json'
    generate_config(json_file, config)
