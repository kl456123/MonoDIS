# import cPickle as pickle

import pickle
import numpy as np
import os

OBJ_CLASSES = ['bg', 'Car']


def read_pkl(pkl_name):
    with open(pkl_name) as f:
        all_dets = pickle.loads(f.read())
    all_dets = np.asarray(all_dets)
    return all_dets


def read_datafile(data_file):
    with open(data_file) as f:
        return f.readlines()


def det2kitti(all_dets, filenames, result_dir='results/data'):
    """
    Args:
        all_dets:(n_classes,n_images,num_dets),
            note that the third dim can varys

    """
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        print("file exist already!\n")
    car_dets = all_dets[1]
    class_name = 'Car'
    for idx, filename in enumerate(filenames):
        det_path = os.path.join(result_dir, filename.strip())
        det_path += '.txt'
        dets_per_img = car_dets[idx]
        with open(det_path, 'w') as f:
            for box in dets_per_img:
                xmin, ymin, xmax, ymax, cf = box
                f.write(
                    '%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n'
                    % (class_name, xmin, ymin, xmax, ymax, cf))


def __save_txt(save_name,
               box_list,
               label_list,
               conf,
               threshold=0.,
               is_none=False):
    file_name = save_name[:-4] + ".txt"
    txt_file = open(file_name, 'w')
    if is_none:
        txt_file.write("None")
        return False

    for bbox, label, cf in zip(box_list, label_list, conf):
        if cf < threshold:
            continue

        xmin, ymin, xmax, ymax = bbox
        # c_x = (bbox[0] + bbox[2]) / 2
        # c_y = (bbox[1] + bbox[3]) / 2
        # if c_x > 0.85 or c_x < 0.15 or c_y > 0.85 or c_y < 0.15:
        #     continue
        class_name = OBJ_CLASSES[label - 1]

        txt_file.write(
            '%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n'
            % (class_name, xmin, ymin, xmax, ymax, cf))


if __name__ == '__main__':
    pkl_name = './results/detections.pkl'
    # val_data_file = '/data/object/training/val.txt'
    val_data_file = './val_car.txt'
    all_dets = read_pkl(pkl_name)
    filenames = read_datafile(val_data_file)
    det2kitti(all_dets, filenames)
