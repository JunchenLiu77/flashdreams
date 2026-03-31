import torch

from flashsim.model.video_dit.alpadreams import (
    CosmosDiT,
    CosmosDiTConfig,
    CosmosDiTCondition,
)


def create_mock_data(
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
    num_views: int = 1,
    len_t: int = 2,
    height: int = 720,
    width: int = 1280,
    encode_with_pixel_shuffle: bool = False,
):
    # tokenizer compression factor
    # temporal_compression_factor = 4
    spatial_compression_factor = 8

    # block latent shape before patchify
    T = len_t
    H = height // spatial_compression_factor
    W = width // spatial_compression_factor

    # Input latent [B, V, T, C, H, W]
    x = torch.randn(
        batch_size,
        num_views,
        T,
        16,  # VAE channels only (condition mask added internally)
        H,
        W,
        device=device,
        dtype=dtype,
    )
    # to make different view has different values
    x = (
        x
        + torch.arange(num_views, device=device, dtype=dtype)[
            None, :, None, None, None, None
        ]
    )

    # Condition mask [B, V, T, 1, H, W] - for first frame conditioning
    condition_video_input_mask = torch.zeros(
        batch_size,
        num_views,
        T,
        1,
        H,
        W,
        device=device,
        dtype=dtype,
    )
    condition_video_input_mask_first_block = condition_video_input_mask.clone()
    condition_video_input_mask_first_block[:, :, :1, :, :, :] = 1.0
    condition_video_input_mask_other_blocks = condition_video_input_mask.clone()

    # Text embeddings [B, V, L, D] - Qwen 7B embeddings (100,352 dims)
    # Using 100,352 dimensions to match the crossattn_projection layer
    text_embeddings = torch.randn(
        batch_size, num_views, 512, 100352, device=device, dtype=dtype
    )

    # Timestep [B]
    timesteps = torch.full((batch_size,), 500.0, device=device, dtype=dtype)

    # HDMap condition [B, V, T, C, H, W]
    if encode_with_pixel_shuffle:
        hdmap_condition = torch.randn(
            batch_size, num_views, T, 192, H, W, device=device, dtype=dtype
        )
    else:
        hdmap_condition = torch.randn(
            batch_size, num_views, T, 16, H, W, device=device, dtype=dtype
        )

    image = torch.randn(batch_size, num_views, 1, 16, H, W, device=device, dtype=dtype)

    data_dict = {
        "x": x,
        "condition_video_input_mask_first_block": condition_video_input_mask_first_block,
        "condition_video_input_mask_other_blocks": condition_video_input_mask_other_blocks,
        "text_embeddings": text_embeddings,
        "timesteps": timesteps,
        "hdmap_condition": hdmap_condition,
        "image": image,
        "T": T,
        "H": H,
        "W": W,
    }
    return data_dict


@torch.no_grad()
def test_alpadreams(
    encode_with_pixel_shuffle: bool = False,
):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    len_t = 4

    data_dict = create_mock_data(
        device=device,
        dtype=dtype,
        batch_size=1,
        num_views=1,
        len_t=len_t,
        height=720,
        width=1280,
        encode_with_pixel_shuffle=encode_with_pixel_shuffle,
    )

    if not encode_with_pixel_shuffle:
        checkpoint_path = "../imaginaire4/data_local/gtc2026_alpamayo_cosmos_cl_demo/checkpoint_cache/32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk2_vae_encode_loc6_gcp.pt"
    else:
        checkpoint_path = None

    config = CosmosDiTConfig(
        len_t=len_t,
        enable_hdmap_condition=True,
        encode_with_pixel_shuffle=encode_with_pixel_shuffle,
        enable_cross_view_attn=False,
        checkpoint_path=checkpoint_path,
    )
    model = CosmosDiT(config=config, dtype=dtype, device=device)

    # patchify the data
    image = data_dict["image"]
    hdmap = data_dict["hdmap_condition"]
    condition_video_input_mask_first_block = data_dict[
        "condition_video_input_mask_first_block"
    ]
    condition_video_input_mask_other_blocks = data_dict[
        "condition_video_input_mask_other_blocks"
    ]
    text_embeddings = data_dict["text_embeddings"]
    H = data_dict["H"]
    W = data_dict["W"]

    cache = model.initialize_cache(
        height=H, width=W, encoded_image=image, text_embeddings=text_embeddings
    )

    condition = CosmosDiTCondition(hdmap, condition_video_input_mask_first_block)
    cache.autoregressive_index = 0
    x0 = model.generate(condition, cache)
    print(x0.shape)

    condition = CosmosDiTCondition(hdmap, condition_video_input_mask_other_blocks)
    cache.autoregressive_index = 1
    x0 = model.generate(condition, cache)
    print(x0.shape)


# python tests/test_alpadreams.py
if __name__ == "__main__":
    test_alpadreams()
