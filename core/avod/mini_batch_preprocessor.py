import numpy as np
import os

from PIL import Image

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.obj_detection import evaluation

from avod.core import box_3d_encoder, anchor_encoder, anchor_filter, anchor_projector
from avod.core import grid_anchor_3d_generator

from avod import top_dir

PRO_ROOT = top_dir()

class MiniBatchPreprocessor(object):
    def __init__(self, dataset):
        """Preprocesses anchors and saves info to files for RPN training

        Args:
            dataset: Dataset object
            mini_batch_dir: directory to save the info
            anchor_strides: anchor strides for generating anchors (per class)
            density_threshold: minimum number of points required to keep an
                anchor
            neg_iou_3d_range: 3D iou range for an anchor to be negative
            pos_iou_3d_range: 3D iou range for an anchor to be positive
        """

        self._dataset = dataset
        self._mini_batch_dir = dataset.mini_batch_save_path
        self._area_extents = dataset.bev_config['area_extents']
        self._anchor_strides = dataset.bev_config['anchor_strides']

        self._density_threshold = dataset.bev_config['density_threshold']
        self._negative_iou_range = dataset.dataset_config['neg_iou_3d_range']
        self._positive_iou_range = dataset.dataset_config['pos_iou_3d_range']

    def _calculate_anchors_info(self,
                                all_anchor_boxes_3d,
                                empty_anchor_filter,
                                gt_labels):
        """Calculates the list of anchor information in the format:
            N x 8 [max_gt_2d_iou, max_gt_3d_iou, (6 x offsets), class_index]
                max_gt_out - highest 3D iou with any ground truth box
                offsets - encoded offsets [dx, dy, dz, d_dimx, d_dimy, d_dimz]
                class_index - the anchor's class as an index
                    (e.g. 0 or 1, for "Background" or "Car")

        Args:
            all_anchor_boxes_3d: list of anchors in box_3d format
                N x [x, y, z, l, w, h, ry]
            empty_anchor_filter: boolean mask of which anchors are non empty
            gt_labels: list of Object Label data format containing ground truth
                labels to generate positives/negatives from.

        Returns:
            list of anchor info
        """
        # Check for ground truth objects
        if len(gt_labels) == 0:
            raise Warning("No valid ground truth label to generate anchors.")

        kitti_utils = self._dataset.kitti_utils

        # Filter empty anchors
        anchor_indices = np.where(empty_anchor_filter)[0]
        anchor_boxes_3d = all_anchor_boxes_3d[empty_anchor_filter]

        # Convert anchor_boxes_3d to anchor format
        anchors = box_3d_encoder.box_3d_to_anchor(anchor_boxes_3d)

        # Convert gt to boxes_3d -> anchors -> iou format
        gt_boxes_3d = np.asarray(
            [box_3d_encoder.object_label_to_box_3d(gt_obj)
             for gt_obj in gt_labels])
        gt_anchors = box_3d_encoder.box_3d_to_anchor(gt_boxes_3d,
                                                     ortho_rotate=True)

        rpn_iou_type = self._dataset.rpn_iou_type
        if rpn_iou_type == '2d':
            # Convert anchors to 2d iou format
            anchors_for_2d_iou, _ = np.asarray(anchor_projector.project_to_bev(
                anchors, kitti_utils.bev_extents))

            gt_boxes_for_2d_iou, _ = anchor_projector.project_to_bev(
                gt_anchors, kitti_utils.bev_extents)

        elif rpn_iou_type == '3d':
            # Convert anchors to 3d iou format for calculation
            anchors_for_3d_iou = box_3d_encoder.box_3d_to_3d_iou_format(
                anchor_boxes_3d)

            gt_boxes_for_3d_iou = \
                box_3d_encoder.box_3d_to_3d_iou_format(gt_boxes_3d)
        else:
            raise ValueError('Invalid rpn_iou_type {}', rpn_iou_type)

        # Initialize sample and offset lists
        num_anchors = len(anchor_boxes_3d)
        all_info = np.zeros((num_anchors, 9))

        # Update anchor indices
        all_info[:, 0] = anchor_indices

        # For each of the labels, generate samples
        for gt_idx in range(len(gt_labels)):

            gt_obj = gt_labels[gt_idx]
            gt_box_3d = gt_boxes_3d[gt_idx]

            # Get 2D or 3D IoU for every anchor
            if rpn_iou_type == '2d':
                gt_box_for_2d_iou = gt_boxes_for_2d_iou[gt_idx]
                ious = evaluation.two_d_iou(gt_box_for_2d_iou,
                                            anchors_for_2d_iou)
            elif rpn_iou_type == '3d':
                gt_box_for_3d_iou = gt_boxes_for_3d_iou[gt_idx]
                ious = evaluation.three_d_iou(gt_box_for_3d_iou,
                                              anchors_for_3d_iou)

            # Only update indices with a higher iou than before
            update_indices = np.greater(
                ious, all_info[:, 1])

            # Get ious to update
            ious_to_update = ious[update_indices]

            # Calculate offsets, use 3D iou to get highest iou
            anchors_to_update = anchors[update_indices]
            gt_anchor = box_3d_encoder.box_3d_to_anchor(gt_box_3d,
                                                        ortho_rotate=True)
            offsets = anchor_encoder.anchor_to_offset(anchors_to_update,
                                                      gt_anchor)

            # Convert gt type to index
            class_idx = kitti_utils.class_str_to_index(gt_obj.type)

            # Update anchors info (indices already updated)
            # [index, iou, (offsets), class_index]
            all_info[update_indices, 1] = ious_to_update

            all_info[update_indices, 2:8] = offsets
            all_info[update_indices, 8] = class_idx

        return all_info

    def preprocess(self, indices):
        """Preprocesses anchor info and saves info to files

        Args:
            indices (int array): sample indices to process.
                If None, processes all samples
        """
        # Get anchor stride for class
        anchor_strides = self._anchor_strides

        dataset = self._dataset
        dataset_utils = self._dataset.kitti_utils
        classes_name = 'Car'

        # Make folder if it doesn't exist yet
        output_dir = self._mini_batch_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get clusters for class
        all_cluster_sizes = np.array([[3.513, 1.581, 1.511], [4.232, 1.653, 1.546]])
        anchor_generator = grid_anchor_3d_generator.GridAnchor3dGenerator()

        # Load indices of data_split
        all_samples = dataset.sample_list

        if indices is None:
            indices = np.arange(len(all_samples))
        num_samples = len(indices)

        # For each image in the dataset, save info on the anchors
        for sample_idx in indices:
            # Get image name for given cluster
            sample_name = all_samples[sample_idx]

            # Check for existing files and skip to the next
            if self._check_for_existing(sample_name):
                print("{} / {}: Sample already preprocessed".format(
                    sample_idx + 1, num_samples, sample_name))
                continue

            ground_truth_list = dataset.get_label(sample_name)
            filtered_gt_list = np.asarray(ground_truth_list)

            # Filtering by class has no valid ground truth, skip this image
            if len(filtered_gt_list) == 0:
                print("{} / {} No {}s for sample {} "
                      "(Ground Truth Filter)".format(
                    sample_idx + 1, num_samples,
                    classes_name, sample_name))

                # Output an empty file and move on to the next image.
                self._save_to_file(classes_name, anchor_strides, sample_name)
                continue

            # Get ground plane
            ground_plane = obj_utils.get_road_plane(sample_name,
                                                    dataset.planes_dir)

            image = Image.fromarray(dataset.get_img(sample_name))
            image_shape = [image.size[1], image.size[0]]

            # Generate sliced 2D voxel grid for filtering
            vx_grid_2d = dataset_utils.create_sliced_voxel_grid_2d(
                sample_name,
                source=dataset.bev_source,
                image_shape=image_shape)

            # List for merging all anchors
            all_anchor_boxes_3d = []

            # Create anchors for each class
            for class_idx in range(len(dataset.classes)):
                # Generate anchors for all classes
                grid_anchor_boxes_3d = anchor_generator._generate(
                    area_3d=self._area_extents,
                    anchor_3d_sizes=all_cluster_sizes,
                    anchor_stride=self._anchor_strides[class_idx],
                    ground_plane=ground_plane)

                all_anchor_boxes_3d.extend(grid_anchor_boxes_3d)

            # Filter empty anchors
            all_anchor_boxes_3d = np.asarray(all_anchor_boxes_3d)
            anchors = box_3d_encoder.box_3d_to_anchor(all_anchor_boxes_3d)
            empty_anchor_filter = anchor_filter.get_empty_anchor_filter_2d(
                anchors, vx_grid_2d, self._density_threshold)

            # Calculate anchor info
            anchors_info = self._calculate_anchors_info(
                all_anchor_boxes_3d, empty_anchor_filter, filtered_gt_list)

            anchor_ious = anchors_info[:, 1]
            valid_iou_indices = np.where(anchor_ious > 0.0)[0]

            print("{} / {}:"
                  "{:>6} anchors, "
                  "{:>6} iou > 0.0, "
                  "for {:>3} {}(s) for sample {}".format(
                sample_idx + 1, num_samples,
                len(anchors_info),
                len(valid_iou_indices),
                len(filtered_gt_list), classes_name, sample_name
            ))

            # Save anchors info
            self._save_to_file(classes_name, anchor_strides,
                               sample_name, anchors_info)

    def _check_for_existing(self, sample_name):
        """
        Checks if a mini batch file exists already

        Args:
            sample_name (str): sample name from dataset, e.g. '000123'

        Returns:
            True if the anchors info file already exists
        """
        return sample_name in os.listdir(self._dataset.mini_batch_save_path)


    def _save_to_file(self, classes_name, anchor_strides, sample_name,
                      anchors_info=np.array([])):
        """
        Saves the anchors info matrix to a file

        Args:
            classes_name (str): classes name, e.g. 'Car', 'Pedestrian',
                'Cyclist', 'People'
            anchor_strides: anchor strides
            sample_name (str): name of sample, e.g. '000123'
            anchors_info: ndarray of anchor info of shape (N, 8)
                N x [index, iou, (6 x offsets), class_index], defaults to
                an empty array
        """

        file_name = os.path.join(self._dataset.mini_batch_save_path, '{}.npy'.format(sample_name))

        # Save to npy file
        anchors_info = np.asarray(anchors_info, dtype=np.float32)
        np.save(file_name, anchors_info)
