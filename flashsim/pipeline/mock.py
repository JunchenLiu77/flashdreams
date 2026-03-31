from dataclasses import dataclass, field

import torch
from torch import Tensor

from flashsim.model.video_vae.mock import (
    MockVideoVAEConfig,
    MockVideoVAE,
    MockVideoVAEEncoderCache,
    MockVideoVAEDecoderCache,
)
from flashsim.model.text_encoder.mock import MockTextEncoderConfig, MockTextEncoder
from flashsim.model.video_dit.mock import (
    MockVideoDiTConfig,
    MockVideoDiT,
    MockVideoDiTCache,
    MockVideoDiTCondition,
)


@dataclass
class MockVideoDiffusionPipelineCache:
    tokenizer_cache: MockVideoVAEEncoderCache
    detokenizer_cache: MockVideoVAEDecoderCache
    dit_cache: MockVideoDiTCache
    dit_condition: MockVideoDiTCondition  # re-usable condition for the video DiT


@dataclass
class MockVideoDiffusionPipelineConfig:
    text_encoder: MockTextEncoderConfig = field(default_factory=MockTextEncoderConfig)
    tokenizer: MockVideoVAEConfig = field(default_factory=MockVideoVAEConfig)
    detokenizer: MockVideoVAEConfig = field(default_factory=MockVideoVAEConfig)
    dit: MockVideoDiTConfig = field(default_factory=MockVideoDiTConfig)


class MockVideoDiffusionPipeline:
    def __init__(
        self,
        config: MockVideoDiffusionPipelineConfig,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cuda"),
    ):
        self.config = config
        self.dtype = dtype
        self.device = device
        self.text_encoder = MockTextEncoder(
            config.text_encoder, dtype=dtype, device=device
        )
        self.tokenizer = MockVideoVAE(config.tokenizer, dtype=dtype, device=device)
        self.dit = MockVideoDiT(config.dit, dtype=dtype, device=device)
        self.detokenizer = MockVideoVAE(config.detokenizer, dtype=dtype, device=device)

    def initialize_cache(
        self, text: list[str], image: Tensor, video_height: int, video_width: int
    ):
        """
        Initialize the cache for the video diffusion pipeline.

        Args:
            text: The batch of texts to encode. [B]
            image: The first frame of the video. [B, 1, 3, H, W]
            video_height: The height of the video.
            video_width: The width of the video.
        """
        encoded_height = video_height // self.tokenizer.spatial_compression_ratio
        encoded_width = video_width // self.tokenizer.spatial_compression_ratio

        encoded_image = self.tokenizer.encode(image)
        encoded_image = self.dit.patchify(encoded_image)
        encoded_text = self.text_encoder.encode(text)
        dit_condition = MockVideoDiTCondition(text=encoded_text, image=encoded_image)
        dit_cache = self.dit.initialize_cache(
            height=encoded_height, width=encoded_width
        )

        tokenizer_cache = self.tokenizer.initialize_encode_cache()
        detokenizer_cache = self.detokenizer.initialize_decode_cache()

        return MockVideoDiffusionPipelineCache(
            tokenizer_cache=tokenizer_cache,
            detokenizer_cache=detokenizer_cache,
            dit_cache=dit_cache,
            dit_condition=dit_condition,
        )

    def streaming_inference(
        self,
        autoregressive_index: int,
        hdmap: Tensor,
        cache: MockVideoDiffusionPipelineCache,
    ):
        """
        Stream the inference of the video diffusion pipeline.

        Args:
            autoregressive_index: The autoregressive index.
            hdmap: The hdmap to encode. [B, T, H, W, D]
            cache: The cache for the video diffusion pipeline.
        """
        # 1. encode the hdmap
        tokenizer_cache = cache.tokenizer_cache
        tokenizer_cache.autoregressive_index = autoregressive_index
        encoded_hdmap = self.tokenizer.encode(hdmap, cache=tokenizer_cache)
        encoded_hdmap = self.dit.patchify(encoded_hdmap)

        # 2. run DiT denoising
        dit_cache = cache.dit_cache
        dit_cache.autoregressive_index = autoregressive_index
        dit_condition = cache.dit_condition
        dit_condition.hdmap = encoded_hdmap
        clean_input = self.dit.generate(condition=dit_condition, cache=dit_cache)
        clean_input = self.dit.unpatchify(clean_input)

        # 3. decode the clean input
        detokenizer_cache = cache.detokenizer_cache
        detokenizer_cache.autoregressive_index = autoregressive_index
        decoded_video = self.detokenizer.decode(clean_input, cache=detokenizer_cache)
        return decoded_video


# python -m flashsim.pipeline.mock
if __name__ == "__main__":
    height = 480
    width = 832

    config = MockVideoDiffusionPipelineConfig()
    pipeline = MockVideoDiffusionPipeline(config)
    cache = pipeline.initialize_cache(
        text=["Hello, world!"],
        image=torch.randn(1, 3, 1, height, width),
        video_height=height,
        video_width=width,
    )

    hdmap = torch.randn(1, 3, 13, height, width)
    decoded_video = pipeline.streaming_inference(
        autoregressive_index=0, hdmap=hdmap, cache=cache
    )
    assert decoded_video.shape == hdmap.shape

    hdmap = torch.randn(1, 3, 16, height, width)
    decoded_video = pipeline.streaming_inference(
        autoregressive_index=1, hdmap=hdmap, cache=cache
    )
    assert decoded_video.shape == hdmap.shape
