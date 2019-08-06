# -*- coding: utf-8 -*-

data_config = {
    'data_root': '/data/kitti',
    'data_type': 'train',
    'bev_config':{
        'height_lo': -0.2,
        'height_hi': 2.3,
        'num_slices': 5,
        'voxel_size': 0.1,
        'area_extents':[[-40, 40], [-5, 3], [0, 70]], # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
        'density_threshold': 1,
    },
    'camera_baseline': 0.54,
    'obj_classes':['Car']

}
