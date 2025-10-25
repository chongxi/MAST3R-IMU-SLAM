import einops
import math
import lietorch
import torch


def as_SE3(X):
    if isinstance(X, lietorch.SE3):
        return X
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )
    T_WC = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC



def yaw_from_sim3(pose: lietorch.Sim3) -> float:
    """Extract planar yaw (rotation about the camera up axis) from a Sim3 pose."""
    mat = pose.matrix()
    if mat.ndim == 3:
        mat = mat[0]
    # Treat the camera forward axis (third column) as the heading direction.
    forward = mat[:3, 2]
    return math.atan2(float(forward[0]), float(forward[2]))
