import torch

from flashdreams.recipes.alpadreams.config import (
    build_sv_2steps_chunk2_loc6_lightvae_lighttae,
)


def test_alpadreams_streaming_inference():
    num_views = 1
    # Must match the alpadreams checkpoint training resolution
    height = 720
    width = 1280

    device = torch.device("cuda")
    dtype = torch.bfloat16

    image = torch.randn(1, num_views, 1, 3, height, width, device=device, dtype=dtype)
    text = [["Hello, world!"] * num_views]

    config = build_sv_2steps_chunk2_loc6_lightvae_lighttae()
    pipeline = config.setup().to(device)
    cache = pipeline.initialize_cache(text=text, image=image)

    autoregressive_index = 0
    num_frames = pipeline.get_num_frames(autoregressive_index)
    hdmap = torch.randn(
        1, num_views, num_frames, 3, height, width, device=device, dtype=dtype
    )
    decoded_video = pipeline.generate(autoregressive_index, hdmap=hdmap, cache=cache)
    pipeline.finalize(autoregressive_index, cache=cache)
    assert decoded_video.shape == hdmap.shape

    autoregressive_index = 1
    num_frames = pipeline.get_num_frames(autoregressive_index)
    hdmap = torch.randn(
        1, num_views, num_frames, 3, height, width, device=device, dtype=dtype
    )
    decoded_video = pipeline.generate(autoregressive_index, hdmap=hdmap, cache=cache)
    pipeline.finalize(autoregressive_index, cache=cache)
    assert decoded_video.shape == hdmap.shape


if __name__ == "__main__":
    test_alpadreams_streaming_inference()
