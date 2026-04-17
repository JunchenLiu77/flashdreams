import torch


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    batch_shape = T.shape[:-2]
    Rot = T[..., :3, :3]  # [..., 3, 3]
    trans = T[..., :3, 3:]  # [..., 3, 1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype).repeat(*batch_shape, 1, 1)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # compute pose between i and i+1
        relative_poses_framewise = torch.bmm(
            SE3_inverse(relative_poses[:-1]), relative_poses[1:]
        )
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:  # note refer to camctrl2: "we scale the coordinate inputs to roughly 1 standard deviation to simplify model learning."
        translations = relative_poses[:, :3, 3]  # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        # only normlaize when moving
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    else:
        max_norm = 1.0
    return relative_poses, max_norm


def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device="cuda",
    dtype=torch.float32,
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias  # [h*w, 2]
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1)  # [f, h*w, 2]
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
    only_rays_d: bool = False,
):
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(
        n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype
    )  # [f, h*w, 2]
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # [f, 1]

    i = grid_xy[..., 0]  # [f, h*w]
    j = grid_xy[..., 1]  # [f, h*w]
    zs = torch.ones_like(i)  # [f, h*w]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)  # [f, h*w, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)  # [f, h*w, 3]

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)  # [f, h*w, 3]
    if only_rays_d:
        plucker_embeddings = rays_d  # [f, h*w, 3]
        plucker_embeddings = plucker_embeddings.view(
            [n_frames, height, width, 3]
        )  # [f*h*w, 3]
    else:
        rays_o = c2ws_mat[:, :3, 3]  # [f, 3]
        rays_o = rays_o[:, None, :].expand_as(rays_d)  # [f, h*w, 3]
        # rays_dxo = torch.cross(rays_o, rays_d, dim=-1) # [f, h*w, 3]
        # note refer to: apt2
        plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)  # [f, h*w, 6]
        plucker_embeddings = plucker_embeddings.view(
            [n_frames, height, width, 6]
        )  # [f*h*w, 6]
    return plucker_embeddings


def compute_relative_poses_causal(
    c2ws_mat: torch.Tensor,  # [..., T, 4, 4]
    trans_normalizer: float = 1.0,
    ref_pose: torch.Tensor | None = None,  # [..., 1, 4, 4]
) -> torch.Tensor:
    if ref_pose is None:
        ref_pose = c2ws_mat[..., 0:1, :, :]
    assert ref_pose.shape[-3:] == (1, 4, 4)
    c2ws_mat = torch.cat([ref_pose, c2ws_mat], dim=-3)
    relative_poses = torch.bmm(
        SE3_inverse(c2ws_mat[..., :-1, :, :]), c2ws_mat[..., 1:, :, :]
    )
    relative_poses[..., :, :3, 3] /= trans_normalizer
    return relative_poses
