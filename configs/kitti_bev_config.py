# -*- coding: utf-8 -*-

# MODEL_CONFIG = {
# 'num_anchors': 6,
# # 'default_ratio': [0.0449, 0.0772, 0.115, 0.164, 0.227],
# 'default_ratio': [0.034, 0.034, 0.034],
# 'aspect_ratio': ((2.39, ), (2.39, )),
# }

# some shared configurations
num_classes = 2
classes = ['bg', 'Car']
class_agnostic = True
bbox_normalize_targets_precomputed = True
bbox_normalize_stds = (0.1, 0.1, 0.2, 0.2)
bbox_normalize_means = (0.0, 0.0, 0.0, 0.0)

train_batch_size = 1
eval_batch_size = 1

# rgb format
normal_mean = [0.485, 0.456, 0.406]
normal_van = [0.229, 0.224, 0.225]

checkpoint_dir = ''

model_config = {
    # 'net': 'resnet50',
    'num_classes': num_classes,
    'output_stride': [4., 8.],
    'input_shape': (800, 700),
    'class_agnostic': class_agnostic,
    # 'pretrained': False,
    'img_channels': 6,
    'classes': classes,
    'rpn_config': {
        'din': 1024,
        'anchor_ratios': [0.5, 1, 2],
        'anchor_scales': [2, 3, 4],
        'feat_stride': 16,
        'pre_nms_topN': 12000,
        'post_nms_topN': 2000,
        'nms_thresh': 0.7,
        'min_size': 16,
        'rpn_clobber_positives': False,
        'rpn_negative_overlap': 0.3,
        'rpn_positive_overlap': 0.5,
        'rpn_batch_size': 512,
        'rpn_fg_fraction': 0.5,
        'rpn_bbox_inside_weights': [1.0, 1.0, 1.0, 1.0],
        'rpn_positive_weight': -1.0,
    },
    'pooling_size': 7,
    'pooling_mode': 'align',
    'crop_resize_with_max_pool': False,
    'truncated': False,
    'proposal_target_layer_config': {
        'nclasses': num_classes,
        'bbox_normalize_means': bbox_normalize_means,
        'bbox_normalize_stds': bbox_normalize_stds,
        'bbox_inside_weights': [1.0, 1.0, 1.0, 1.0],
        'batch_size': 512,
        'fg_fraction': 0.25,
        'bbox_normalize_targets_precomputed':
        bbox_normalize_targets_precomputed,
        'fg_thresh': 0.5,
        'bg_thresh': 0.5,
        'bg_thresh_lo': 0.0,
    },
}

data_config = {
    'name': 'kitti',
    'dataset_config': {
        'root_path': '/data/object/training',
        'dataset_file': 'train.txt',
        'bev_config': {
            'height_lo': -0.2,
            'height_hi': 2.3,
            'num_slices': 5,
            'voxel_size': 0.1,
            'area_extents':
            [[-40, 40], [-5, 3],
             [0, 70]],  # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            'density_threshold': 1,
        },
        'camera_baseline': 0.54
    },
    'transform_config': {},
    'dataloader_config': {
        'shuffle': True,
        'batch_size': train_batch_size,
        'num_workers': 1
    }
}

eval_data_config = {
    'name': 'kitti',
    'dataset_config': {
        'root_path': '/data/object/training',
        'dataset_file': 'val.txt',
        'cache_bev': True,
        'bev_config': {
            'height_lo': -0.2,
            'height_hi': 2.3,
            'num_slices': 5,
            'voxel_size': 0.1,
            'area_extents':
            [[-40, 40], [-5, 3],
             [0, 70]],  # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            'density_threshold': 1,
        },
        'camera_baseline': 0.54
    },
    'transform_config': {},
    'dataloader_config': {
        'shuffle': False,
        'batch_size': eval_batch_size,
        'num_workers': 1
    },
}

train_config = {
    'rng_seed': 3,
    'save_dir': checkpoint_dir,
    'device_ids': [0],
    'disp_interval': 100,
    'max_epochs': 100,
    'checkpoint_interval': 10000,
    'mGPUs': True,
    'clip_gradient': 10,
    'start_epoch': 1,
    'scheduler_config': {
        'type': 'step',
        'lr_decay_gamma': 0.1,
        'lr_decay_step': 20,
        'last_epoch': -1,
    },
    'optimizer_config': {
        'type': 'sgd',
        'momentum': 0.9,
        'lr': 0.001,
    }
}

eval_config = {
    'rng_seed': 3,
    'cache_img_dir': './results/bev',
    'cache_gt_dir': './results/label',
    'load_dir': checkpoint_dir,
    'max_per_image': 100,
    'bbox_reg': True,
    'bbox_normalize_targets_precomputed': bbox_normalize_targets_precomputed,
    'bbox_normalize_means': bbox_normalize_means,
    'bbox_normalize_stds': bbox_normalize_stds,
    'batch_size': eval_batch_size,
    'class_agnostic': True,
    'thresh': 0.5,
    'nms': 0.3,
    'classes': classes,
    'eval_out': './results/data',
}
