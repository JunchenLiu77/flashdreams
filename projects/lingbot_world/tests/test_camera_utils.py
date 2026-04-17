import torch

import numpy as np
from projects.lingbot_world.camera_utils import (
    compute_relative_poses,
    compute_relative_poses_causal,
)


def test_compute_relative_poses_causal():
    camera_path = "assets/example_data/lingbot_world/poses.npy"
    poses = torch.from_numpy(np.load(camera_path)).float()

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


# python -m projects.lingbot_world.tests.test_camera_utils
if __name__ == "__main__":
    test_compute_relative_poses_causal()
