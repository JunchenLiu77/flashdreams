from dataclasses import dataclass

import torch
from torch import Tensor

from flashsim.model.video_vae.wan import WanVAEInterface, WanVAECache
from flashsim.model.text_encoder.cosmos_reason1 import CosmosReason1TextEncoder
from flashsim.model.video_dit.alpadreams.model import (
    CosmosDiT,
    CosmosDiTCache,
    CosmosDiTCondition,
    CosmosDiTConfig,
)


@dataclass
class AlpadreamsPipelineCache:
    tokenizer_cache: WanVAECache
    detokenizer_cache: WanVAECache
    dit_cache: CosmosDiTCache


class AlpadreamsPipeline:
    def __init__(
        self,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        self.dtype = dtype
        self.device = device
        self.text_encoder = CosmosReason1TextEncoder(device=device)
        self.detokenizer = self.tokenizer = WanVAEInterface(
            checkpoint_path="../imaginaire4/data_local/gtc2026_alpamayo_cosmos_cl_demo/Autoencoders/Wan2.1_VAE.safetensors",
            use_lightvae=False,
            dtype=dtype,
            device=device,
        )

        dit_config = CosmosDiTConfig(
            len_t=3,
            window_size_t=6,
            enable_hdmap_condition=True,
            encode_with_pixel_shuffle=False,
            enable_cross_view_attn=False,
            checkpoint_path="../imaginaire4/data_local/gtc2026_alpamayo_cosmos_cl_demo/checkpoint_cache/32n_cosmos_v2_2b_SF_res720p_30fps_i2v_hdmap_chunk3_vae_encode_loc6_gcp.pt",
            denoising_timesteps=(1000, 450),
        )
        self.dit = CosmosDiT(config=dit_config, dtype=dtype, device=device)

    def initialize_cache(
        self, text: list[list[str]], image: Tensor, view_names: list[str] | None = None
    ):
        """
        Initialize the cache for the Alpadreams pipeline.

        Args:
            text: The batch of texts to encode. [B, V]
            image: The first frame of the video. [B, V, 1, 3, H, W]
        """
        video_height, video_width = image.shape[-2:]

        encoded_height = video_height // self.tokenizer.spatial_compression_ratio
        encoded_width = video_width // self.tokenizer.spatial_compression_ratio

        image_embedding = self.tokenizer.encode(image)
        text_embeddings = torch.stack(
            [self.text_encoder.encode(t) for t in text], dim=0
        )

        dit_cache = self.dit.initialize_cache(
            height=encoded_height,
            width=encoded_width,
            encoded_image=image_embedding,
            text_embeddings=text_embeddings,
            view_names=view_names,
        )

        tokenizer_cache = self.tokenizer.initialize_encode_cache()
        detokenizer_cache = self.detokenizer.initialize_decode_cache()

        return AlpadreamsPipelineCache(
            tokenizer_cache=tokenizer_cache,
            detokenizer_cache=detokenizer_cache,
            dit_cache=dit_cache,
        )

    @torch.no_grad()
    def streaming_inference(
        self, autoregressive_index: int, hdmap: Tensor, cache: AlpadreamsPipelineCache
    ):
        """
        Stream the inference of the video diffusion pipeline.

        Args:
            autoregressive_index: The autoregressive index.
            hdmap: The hdmap to encode. [B, V, T, C, H, W]
            cache: The cache for the Alpadreams pipeline.
        """
        # 1. encode the hdmap
        encoded_hdmap = self.tokenizer.encode(hdmap, cache=cache.tokenizer_cache)
        B, V, T, C, H, W = encoded_hdmap.shape

        # 2. run DiT denoising
        cache.dit_cache.autoregressive_index = autoregressive_index
        clean_input = self.dit.generate(
            condition=CosmosDiTCondition(hdmap=encoded_hdmap), cache=cache.dit_cache
        )

        # 3. decode the clean input
        decoded_video = self.detokenizer.decode(
            clean_input, cache=cache.detokenizer_cache
        )
        return decoded_video

    @torch.no_grad()
    def finalize(self, cache: AlpadreamsPipelineCache) -> None:
        """
        Finalize the streaming inference. This will update the KV cache for the next block.
        """
        self.dit.finalize(cache.dit_cache)

    @torch.no_grad()
    def get_num_frames(self, autoregressive_index: int) -> int:
        """
        Get the number of frames for the given autoregressive index.
        """
        if autoregressive_index == 0:
            return (
                1
                + (self.dit.config.len_t - 1)
                * self.detokenizer.temporal_compression_ratio
            )
        else:
            return self.dit.config.len_t * self.detokenizer.temporal_compression_ratio


# python -m flashsim.pipeline.alpadreams
if __name__ == "__main__":
    num_views = 1
    height = 704
    width = 1280

    device = torch.device("cuda")
    dtype = torch.bfloat16

    image = torch.randn(1, num_views, 1, 3, height, width, device=device, dtype=dtype)
    text = [["Hello, world!"] * num_views]

    pipeline = AlpadreamsPipeline(dtype=dtype, device=device)
    cache = pipeline.initialize_cache(text=text, image=image)

    hdmap = torch.randn(1, num_views, 5, 3, height, width, device=device, dtype=dtype)
    decoded_video = pipeline.streaming_inference(
        autoregressive_index=0, hdmap=hdmap, cache=cache
    )
    assert decoded_video.shape == hdmap.shape

    hdmap = torch.randn(1, num_views, 8, 3, height, width, device=device, dtype=dtype)
    decoded_video = pipeline.streaming_inference(
        autoregressive_index=1, hdmap=hdmap, cache=cache
    )
    assert decoded_video.shape == hdmap.shape
