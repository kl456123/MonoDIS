import torch

def torch_orientation_to_angle_vector(orientations_tensor):
    """ Converts orientation angles into angle unit vector representation.
        e.g. 45 -> [0.717, 0.717], 90 -> [0, 1]

    Args:
        orientations_tensor: A tensor of shape (N,) of orientation angles

    Returns:
        A tensor of shape (N, 2) of angle unit vectors in the format [x, y]
    """
    x = torch.cos(orientations_tensor)
    y = torch.sin(orientations_tensor)

    return torch.stack([x, y], dim=1)


def torch_angle_vector_to_orientation(angle_vectors_tensor):
    """ Converts angle unit vectors into orientation angle representation.
        e.g. [0.717, 0.717] -> 45, [0, 1] -> 90

    Args:
        angle_vectors_tensor: a tensor of shape (N, 2) of angle unit vectors
            in the format [x, y]

    Returns:
        A tensor of shape (N,) of orientation angles
    """
    x = angle_vectors_tensor[:, 0]
    y = angle_vectors_tensor[:, 1]

    return torch.atan2(y, x)
