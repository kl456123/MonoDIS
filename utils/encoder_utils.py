# -*- coding: utf-8 -*-

from utils import geometry_utils
from core.utils import format_checker
import torch


def encode_lines(lines, proposals):
    """
    Args:
        lines: shape(N, 2, 2)
    """
    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
        proposals.unsqueeze(0))[0]
    encoded_lines = (
        lines - proposals_xywh[:, None, :2]) / proposals_xywh[:, None, 2:]
    return encoded_lines


def encode_points(points, proposals):
    """
    Args:
        points: shape(N, 2)
        proposals: shape(N, 4)
    """
    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
        proposals.unsqueeze(0))[0]
    encoded_points = (points - proposals_xywh[:, :2]) / proposals_xywh[:, 2:]
    return encoded_points


def decode_points(encoded_points, proposals):
    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
        proposals.unsqueeze(0))[0]
    points = encoded_points * proposals_xywh[:, 2:] + proposals_xywh[:, :2]
    return points


def decode_lines(encoded_lines, proposals):
    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
        proposals.unsqueeze(0))[0]
    lines = encoded_lines.view(
        -1, 2, 2) * proposals_xywh[:, None, 2:] + proposals_xywh[:, None, :2]
    return lines.view(-1, 4)


def encode_ray(lines, proposals):
    format_checker.check_tensor_shape(lines, [None, 2, 2])
    encoded_points = encode_points(lines[:, 0], proposals)

    direction = lines[:, 0] - lines[:, 1]
    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
        proposals.unsqueeze(0))[0]
    # pooling_size should be the same in x and y direction
    normalized_direction = direction / proposals_xywh[:, 2:]
    norm = torch.norm(normalized_direction, dim=-1)
    cos = normalized_direction[:, 0] / norm
    sin = normalized_direction[:, 1] / norm
    normalized_direction = torch.stack([cos, sin], dim=-1)
    # theta = torch.atan2(normalized_direction[:, 1], normalized_direction[:, 0])
    encoded_lines = torch.cat([encoded_points, normalized_direction], dim=-1)
    return encoded_lines


def decode_ray(encoded_lines, proposals, p2):
    format_checker.check_tensor_shape(encoded_lines, [None, 4])
    format_checker.check_tensor_shape(proposals, [None, 4])
    encoded_points = encoded_lines[:, :2]

    normalized_direction = encoded_lines[:, 2:]
    norm = torch.norm(normalized_direction, dim=-1)
    cos = normalized_direction[:, 0] / norm
    sin = normalized_direction[:, 1] / norm
    normalized_direction = torch.stack([cos, sin], dim=-1)

    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(proposals.unsqueeze(0))[0]
    deltas = normalized_direction * proposals_xywh[:, 2:]
    points1 = decode_points(encoded_points, proposals)
    points2 = points1 - deltas

    lines = torch.cat([points1, points2], dim=-1)
    ry = geometry_utils.torch_pts_2d_to_dir_3d(
        lines.unsqueeze(0), p2.unsqueeze(0))[0].unsqueeze(-1)
    return ry


def decode_ry(encoded_lines, proposals, p2):
    lines = decode_lines(encoded_lines, proposals)

    ry = geometry_utils.torch_pts_2d_to_dir_3d(
        lines.unsqueeze(0), p2.unsqueeze(0))[0].unsqueeze(-1)
    return ry
