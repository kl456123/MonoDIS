# -*- coding: utf-8 -*-

import os
import argparse

label_dir = '/data/object/training/label_2/'
result_dir = './results/data'

output_dir = './results/analysis'
case_types = ['fps', 'fns']

#  def generate_val_sets(result_dir):
#  sort(os.listdir(result_dir)):


def parse_fn(fn):
    fn = os.path.splitext(os.path.basename(fn))[0]
    items = fn.split("_")
    thresh_ind = items[0]
    case_type = items[1]
    if case_type == 'fns':
        pass
    elif case_type == 'fps':
        pass
    else:
        raise ValueError("filename {} can not be parsed".format(fn))
    return thresh_ind, case_type


def parse_file_to_lines(box_fn_path):
    with open(box_fn_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        return lines


def parse_line(line):
    """
    Args:
        line: a line that needs to be parsed
        bbox_dir: directory that contains files of bbox info
        case_type: indicates 'fns' or 'fps'
    """
    #  if case_type == 'fns':
    #  pass
    #  elif case_type == 'fps':
    #  pass
    items = line.split(" ")
    img_ind = int(items[0])
    xmin = float(items[1])
    ymin = float(items[2])
    xmax = float(items[3])
    ymax = float(items[4])

    # generate bbox file name
    #  bbox_fn_path = os.path.join(bbox_dir, "{:06}.txt".format(img_ind))
    #  lines = parse_file_to_lines(box_fn_path)
    #  boxes_info = []
    #  return bbox_fn_path
    box_fn = "{:06}.txt".format(img_ind)
    return box_fn, [xmin, ymin, xmax, ymax]


def parse_dict(name2boxes, box_dir):
    """
        map box_fn_path with boxes
    """
    path2boxes = {}
    for box_fn, boxes_ind in name2boxes.items():
        box_fn_path = os.path.join(box_dir, box_fn)
        lines = parse_file_to_lines(box_fn_path)
        boxes = []
        for ind in boxes_ind:
            boxes.append(lines[ind])

        path2boxes[box_fn] = boxes
    return path2boxes


def parse_file_to_dict(fn_path, box_dir):
    lines = parse_file_to_lines(fn_path)

    name2boxes = {}
    for line in lines:
        box_fn, boxes = parse_line(line)
        if name2boxes.get(box_fn):
            name2boxes[box_fn].append(boxes)
        else:
            name2boxes[box_fn] = [boxes]

    # return parse_dict(name2boxes, box_dir)
    return name2boxes


def boxes2kitti(boxes):
    class_name = 'Car'
    cf = 1.0
    boxes_str = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        boxes_str.append(
            '%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.3f\n'
            % (class_name, xmin, ymin, xmax, ymax, cf))
    return boxes_str


def generate_file_from_dict(parsed_dict, output_dir):
    os.makedirs(output_dir)
    for img_fn, boxes in parsed_dict.items():
        parsed_fn_path = os.path.join(output_dir, img_fn)
        with open(parsed_fn_path, 'w') as f:
            boxes_str = boxes2kitti(boxes)
            f.write('\n'.join(boxes_str))


def postprocess(fn_path, output_dir=None):
    # analysis file name first
    thresh_ind, case_type = parse_fn(fn_path)

    if case_type == 'fns':
        box_dir = label_dir
    else:
        box_dir = result_dir

    # analysis file
    parsed_dict = parse_file_to_dict(fn_path, box_dir)
    if output_dir is not None:
        # dump to pickle file
        # if os.path.isdir(os.path.join(output_dir, case_type)):
        # print("output dir {} has exist already!".format(output_dir))
        # import ipdb
        # ipdb.set_trace()
        # os.system("rm {} -rf".format(output_dir))
        output_dir = os.path.join(output_dir, case_type, str(thresh_ind))
        generate_file_from_dict(parsed_dict, output_dir)
    return parsed_dict


def postprocess_all(analysis_dir, output_dir):
    for file in os.listdir(analysis_dir):
        fn_path = os.path.join(analysis_dir, file)
        if os.path.isfile(fn_path):
            if os.path.splitext(fn_path)[-1] == ".txt":
                postprocess(fn_path, output_dir)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--fn", dest="fn", help="path to file that needs to be parsed")
parser.add_argument("--dir", dest="dir", help="path to analysis_dir")

if __name__ == '__main__':
    args = parser.parse_args()
    for case_type in case_types:
        output_path = os.path.join(output_dir, case_type)
        if os.path.isdir(output_path):
            print("output dir {} has exist already!".format(output_dir))
            import ipdb
            ipdb.set_trace()
            os.system("rm {} -rf".format(output_path))
        os.makedirs(output_path)

    if args.dir is not None:
        args.dir = './results/analysis'
    postprocess_all(args.dir, output_dir)
    #  else:
#  postprocess(args.fn, output_dir)
