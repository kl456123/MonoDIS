# -*- coding: utf-8 -*-

import os
import warnings

analysis_dir = './results/analysis'
label_dir = '/data/object/training/label_2'
result_dir = './results/data'

DIFFICULTY = ['moderate']
FALSE_CASES = ['fns', 'fns_bst', 'fps', 'fps_bst']

# mask


def checkDuplicate(sample_idx, det_idx, sets):
    hash_id = str(sample_idx) + '_' + str(det_idx)
    if hash_id in sets:
        return True
    else:
        sets.add(hash_id)
        return False


def checkFileType(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split(' ')
            if len(line) == 1:
                return 0
            else:
                return 1


def handle(path, use_dir, sets):
    sample_fn = os.path.basename(path)
    sample_idx = sample_fn[:-4]
    idxs = []
    # load idx from analysis dir
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            idxs.append(int(line))

    # load boxes from use_dir

    boxes_str = []
    use_path = os.path.join(use_dir, sample_fn)
    with open(use_path, 'r') as f:
        for line in f.readlines():
            boxes_str.append(line.strip())

    # select according to idx
    # deduplicated
    dedup_boxes_str = []
    for box_id in idxs:
        if not checkDuplicate(sample_idx, box_id, sets):
            if box_id < len(boxes_str):
                dedup_boxes_str.append(boxes_str[box_id])
            else:
                warnings.warn("some errors happened, ignored here!")

    # nothing need to be saved,remove it directly
    if len(dedup_boxes_str) == 0:
        os.remove(path)
    else:
        # save to file
        # override files in analysis dir
        with open(path, 'w') as f:
            f.write('\n'.join(dedup_boxes_str))


def listDir(dir_path):
    # sorted in order
    for file in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            yield file_path
        elif os.path.isdir(file_path):
            for file_path in listDir(file_path):
                yield file_path


def main(analysis_dir):
    for difficulty in DIFFICULTY:
        for false_case in FALSE_CASES:
            sets = set()
            if false_case == 'fns' or false_case == 'fns_bst':
                use_dir = label_dir
            else:
                use_dir = result_dir
            dirpath = os.path.join(
                os.path.join(analysis_dir, difficulty), false_case)
            for path in listDir(dirpath):
                # if empty,remove it
                if os.stat(path).st_size == 0:
                    os.remove(path)
                    continue
                file_type = checkFileType(path)
                # 0 means not finished
                # 1 means finished
                if file_type == 0:

                    handle(path, use_dir, sets)
                elif file_type == 1:
                    print('{} is finished already!'.format(path))


if __name__ == '__main__':
    main(analysis_dir)
