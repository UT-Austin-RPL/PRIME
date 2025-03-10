import torch
import numpy as np
import math

PI = np.pi
EPS = np.finfo(float).eps * 4.0

def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    if torch.is_tensor(quaternion):
        inds = torch.tensor([3, 0, 1, 2]).to(quaternion.device)
        q = quaternion[inds]

        n = torch.dot(q, q)
        if n < EPS:
            return torch.eye(3).to(quaternion.device)
        q *= math.sqrt(2.0 / n)
        q2 = torch.outer(q, q)
        return torch.tensor(
            [
                [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
                [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
                [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
            ]
        ).to(quaternion.device)
    else:
        # awkward semantics for use with numba
        inds = np.array([3, 0, 1, 2])
        q = np.asarray(quaternion).copy().astype(np.float32)[inds]

        n = np.dot(q, q)
        if n < EPS:
            return np.identity(3)
        q *= math.sqrt(2.0 / n)
        q2 = np.outer(q, q)
        return np.array(
            [
                [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
                [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
                [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
            ]
        )