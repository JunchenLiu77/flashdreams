from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashsim.model.video_vae.wan import WanVAEInterfaceConfig, WanVAECache
from flashsim.model.video_vae.teahv import TeahvInterfaceConfig, TAEHVCache
from flashsim.model.text_encoder.cosmos_reason1 import CosmosReason1TextEncoderConfig
from flashsim.model.video_dit.alpadreams.model import (
    CosmosDiTCache,
    CosmosDiTCondition,
    CosmosDiTConfig,
)
from flashsim.configs import InstantiateConfig


@dataclass
class AlpadreamsPipelineCache:
    tokenizer_cache: WanVAECache | TAEHVCache
    detokenizer_cache: WanVAECache | TAEHVCache
    dit_cache: CosmosDiTCache


@dataclass
class AlpadreamsPipelineConfig(InstantiateConfig["AlpadreamsPipeline"]):
    _target: type["AlpadreamsPipeline"] = field(
        default_factory=lambda: AlpadreamsPipeline
    )

    tokenizer: WanVAEInterfaceConfig | TeahvInterfaceConfig = field(
        default_factory=lambda: WanVAEInterfaceConfig()
    )
    detokenizer: WanVAEInterfaceConfig | TeahvInterfaceConfig = field(
        default_factory=lambda: TeahvInterfaceConfig()
    )
    text_encoder: CosmosReason1TextEncoderConfig = field(
        default_factory=lambda: CosmosReason1TextEncoderConfig()
    )
    image_encoder: WanVAEInterfaceConfig | TeahvInterfaceConfig = field(
        default_factory=lambda: WanVAEInterfaceConfig()
    )
    dit: CosmosDiTConfig = field(default_factory=lambda: CosmosDiTConfig())


class AlpadreamsPipeline:
    def __init__(
        self,
        config: AlpadreamsPipelineConfig,
        device: torch.device = torch.device("cuda"),
    ):
        self.text_encoder = config.text_encoder.setup(device=device)
        self.image_encoder = config.image_encoder.setup(device=device)
        self.tokenizer = config.tokenizer.setup(device=device)
        self.detokenizer = config.detokenizer.setup(device=device)
        self.dit = config.dit.setup(device=device)

    def initialize_cache(
        self, text: list[list[str]], image: Tensor, view_names: list[str] | None = None
    ) -> AlpadreamsPipelineCache:
        """
        Initialize the cache for the Alpadreams pipeline.

        Args:
            text: The batch of texts to encode. [B, V]
            image: The first frame of the video. [B, V, 1, 3, H, W]
        """
        video_height, video_width = image.shape[-2:]

        encoded_height = video_height // self.tokenizer.spatial_compression_ratio
        encoded_width = video_width // self.tokenizer.spatial_compression_ratio

        image_embedding = self.image_encoder.encode(image)
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
    ) -> Tensor:
        """
        Stream the inference of the video diffusion pipeline.

        Args:
            autoregressive_index: The autoregressive index.
            hdmap: The hdmap to encode. [B, V, T, C, H, W]
            cache: The cache for the Alpadreams pipeline.

        Returns:
            The decoded video. [B, V, T, C, H, W]
        """
        # 1. encode the hdmap
        if hasattr(cache.tokenizer_cache, "autoregressive_index"):
            cache.tokenizer_cache.autoregressive_index = autoregressive_index
        encoded_hdmap = self.tokenizer.encode(hdmap, cache=cache.tokenizer_cache)

        # 2. run DiT denoising
        cache.dit_cache.autoregressive_index = autoregressive_index
        clean_input = self.dit.generate(
            condition=CosmosDiTCondition(hdmap=encoded_hdmap), cache=cache.dit_cache
        )

        # 3. decode the clean input
        if hasattr(cache.detokenizer_cache, "autoregressive_index"):
            cache.detokenizer_cache.autoregressive_index = autoregressive_index
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
