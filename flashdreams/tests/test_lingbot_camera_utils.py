import torch

from flashdreams.recipes.lingbot_world.encoder.utils import (
    compute_relative_poses,
    compute_relative_poses_causal,
)


def random_SO3(batch_size: tuple[int], device="cpu"):
    # Step 1: Generate a batch of random matrices of shape (batch_size, 3, 3)
    random_matrices = torch.randn((*batch_size, 3, 3), device=device)
    random_matrices = random_matrices.reshape(-1, 3, 3)

    # Step 2: Apply QR decomposition to each matrix in the batch
    # The `torch.linalg.qr` function works for batches of matrices in newer PyTorch versions
    q, r = torch.linalg.qr(random_matrices)
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[..., None, :]

    # Step 3: Adjust for positive determinant in each matrix
    # Compute the determinants and find indices where the determinant is negative
    det_q = torch.det(q)
    negative_det_indices = det_q < 0

    # Flip the sign of the last column where determinant is negative
    q[negative_det_indices, :, 2] *= -1
    q = q.reshape(*batch_size, 3, 3)

    return q


def random_SE3(batch_size: tuple[int], device="cpu"):
    random_matrices = torch.eye(4, device=device).repeat(*batch_size, 1, 1)
    random_matrices[..., :3, :3] = random_SO3(batch_size, device)
    random_matrices[..., :3, 3] = torch.randn(*batch_size, 3, device=device)
    return random_matrices


def test_compute_relative_poses_causal():
    poses = random_SE3((10,))

    relative_poses1, trans_normalizer = compute_relative_poses(poses, framewise=True)
    relative_poses2 = compute_relative_poses_causal(poses, trans_normalizer)
    torch.testing.assert_close(relative_poses1, relative_poses2, atol=1e-4, rtol=1e-4)

    last_pose = None
    relative_poses3 = []
    for pose in poses:
        pose = pose.unsqueeze(0)
        relative_pose = compute_relative_poses_causal(pose, trans_normalizer, last_pose)
        relative_poses3.append(relative_pose)
        last_pose = pose
    relative_poses3 = torch.cat(relative_poses3, dim=0)
    torch.testing.assert_close(relative_poses1, relative_poses3, atol=1e-4, rtol=1e-4)


# python -m tests.test_camera_utils
if __name__ == "__main__":
    test_compute_relative_poses_causal()
