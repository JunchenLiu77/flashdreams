from dataclasses import dataclass, field
from loguru import logger

import torch
from torch import Tensor

from flashsim.model.video_vae.wan import WanVAEInterfaceConfig, WanVAECache
from flashsim.model.video_vae.teahv import TeahvInterfaceConfig, TAEHVCache
from flashsim.model.text_encoder.wan2_1 import WanTextEncoderConfig
from flashsim.model.video_dit.wan2_2.model import (
    WanDiTCache,
    WanDiTCondition,
    WanDiTConfig,
)
from flashsim.configs import InstantiateConfig


class ProfileEvents:
    def __init__(self):
        # sequential events
        self.tic = torch.cuda.Event(enable_timing=True)
        self.toc_after_encode = torch.cuda.Event(enable_timing=True)
        self.toc_after_denoise = torch.cuda.Event(enable_timing=True)
        self.toc_after_decode = torch.cuda.Event(enable_timing=True)
        self.toc_after_finalize = torch.cuda.Event(enable_timing=True)

    def summary(self) -> dict[str, float]:
        return {
            "elapsed_time_encode": self.tic.elapsed_time(self.toc_after_encode),
            "elapsed_time_denoise": self.toc_after_encode.elapsed_time(
                self.toc_after_denoise
            ),
            "elapsed_time_decode": self.toc_after_denoise.elapsed_time(
                self.toc_after_decode
            ),
            "elapsed_time_finalize": self.toc_after_decode.elapsed_time(
                self.toc_after_finalize
            ),
            "time_to_decode": self.tic.elapsed_time(self.toc_after_decode),
            "time_to_finalize": self.tic.elapsed_time(self.toc_after_finalize),
        }

    @staticmethod
    def finalize(events: list["ProfileEvents"], skip_first_n: int = 0) -> None:
        if skip_first_n > 0:
            events = events[skip_first_n:]

        n = len(events)

        ts = []
        for event in events:
            ts.append(event.summary())

        elapsed_time_encode = sum(t["elapsed_time_encode"] for t in ts)
        elapsed_time_denoise = sum(t["elapsed_time_denoise"] for t in ts)
        elapsed_time_decode = sum(t["elapsed_time_decode"] for t in ts)
        elapsed_time_finalize = sum(t["elapsed_time_finalize"] for t in ts)
        time_to_decode = sum(t["time_to_decode"] for t in ts)
        time_to_finalize = sum(t["time_to_finalize"] for t in ts)

        def perc1(t):
            return f"({t / time_to_decode * 100:06.3f}%)"

        logger.info(
            f"Profiling results for {n} events after skipping first {skip_first_n} events:"
        )
        logger.info(f"Average Latency to Decode: {time_to_decode / n / 1000.0} seconds")
        logger.info(
            f"   ├─{perc1(elapsed_time_encode)} VAE encode HD map {elapsed_time_encode / n:.4f} ms"
        )
        logger.info(
            f"   ├─{perc1(elapsed_time_denoise)} DiT denoise latent {elapsed_time_denoise / n:.4f} ms"
        )
        logger.info(
            f"   ╰─{perc1(elapsed_time_decode)} VAE decode {elapsed_time_decode / n:.4f} ms"
        )
        logger.info(
            f"Average Latency to Finalize: {time_to_finalize / n / 1000.0} seconds"
        )
        logger.info(f"   ╰─finalize KV cache {elapsed_time_finalize / n:.4f} ms")


@dataclass
class Wan2_2PipelineCache:
    tokenizer_cache: WanVAECache | TAEHVCache
    detokenizer_cache: WanVAECache | TAEHVCache
    dit_cache: WanDiTCache
    profile_events: list[ProfileEvents]


@dataclass
class Wan2_2PipelineConfig(InstantiateConfig["Wan2_2Pipeline"]):
    _target: type["Wan2_2Pipeline"] = field(default_factory=lambda: Wan2_2Pipeline)

    tokenizer: WanVAEInterfaceConfig | TeahvInterfaceConfig = field(
        default_factory=lambda: WanVAEInterfaceConfig()
    )
    detokenizer: WanVAEInterfaceConfig | TeahvInterfaceConfig = field(
        default_factory=lambda: TeahvInterfaceConfig()
    )
    text_encoder: WanTextEncoderConfig = field(
        default_factory=lambda: WanTextEncoderConfig()
    )
    image_encoder: WanVAEInterfaceConfig | TeahvInterfaceConfig = field(
        default_factory=lambda: WanVAEInterfaceConfig()
    )
    dit: WanDiTConfig = field(default_factory=lambda: WanDiTConfig())

    seed: int = 42


class Wan2_2Pipeline:
    def __init__(
        self,
        config: Wan2_2PipelineConfig,
        device: torch.device = torch.device("cuda"),
    ):
        self.text_encoder = config.text_encoder.setup(device=device)
        self.image_encoder = config.image_encoder.setup(device=device)
        self.tokenizer = config.tokenizer.setup(device=device)
        self.detokenizer = config.detokenizer.setup(device=device)
        self.dit = config.dit.setup(device=device)
        self.rng = torch.Generator(device=device).manual_seed(config.seed)

    def initialize_cache(
        self,
        video_height: int,
        video_width: int,
        text: list[list[str]],
        image: Tensor | None = None,
    ) -> Wan2_2PipelineCache:
        """
        Initialize the cache for the Wan2_2 pipeline.

        Args:
            text: The batch of texts to encode. [B, V]
            image: The first frame of the video. [B, V, 1, 3, H, W] or None for text-to-video
        """
        encoded_height = video_height // self.tokenizer.spatial_compression_ratio
        encoded_width = video_width // self.tokenizer.spatial_compression_ratio

        if image is not None:
            assert image.shape[-2:] == (video_height, video_width), (
                f"image shape must be {video_height}x{video_width}, but got {image.shape[-2:]}"
            )
            initial_latent = self.image_encoder.encode(image)
        else:
            initial_latent = None

        text_embeddings = torch.stack(
            [self.text_encoder.encode(t) for t in text], dim=0
        )

        dit_cache = self.dit.initialize_cache(
            height=encoded_height,
            width=encoded_width,
            text_embeddings=text_embeddings,
            initial_latent=initial_latent,
        )

        tokenizer_cache = self.tokenizer.initialize_encode_cache()
        detokenizer_cache = self.detokenizer.initialize_decode_cache()

        # if the initial latent is available, refresh the cache with it.
        if initial_latent is not None:
            _ = self.detokenizer.decode(initial_latent, cache=detokenizer_cache)
            _ = self.dit.finalize(dit_cache, context_noise=0.0, rng=self.rng)

        return Wan2_2PipelineCache(
            tokenizer_cache=tokenizer_cache,
            detokenizer_cache=detokenizer_cache,
            dit_cache=dit_cache,
            profile_events=[],
        )

    @torch.no_grad()
    def streaming_inference(
        self,
        autoregressive_index: int,
        cache: Wan2_2PipelineCache,
    ) -> Tensor:
        """
        Stream the inference of the video diffusion pipeline.

        Args:
            autoregressive_index: The autoregressive index.
            cache: The cache for the Wan2_2 pipeline.

        Returns:
            The decoded video. [B, V, T, C, H, W]
        """
        if autoregressive_index >= len(cache.profile_events):
            cache.profile_events.append(ProfileEvents())
        profile_events = cache.profile_events[-1]

        if profile_events is not None:
            profile_events.tic.record()

        if profile_events is not None:
            profile_events.toc_after_encode.record()

        # 2. run DiT denoising
        assert autoregressive_index > cache.dit_cache.autoregressive_index, (
            f"Autoregressive index must be greater than the current "
            f"autoregressive index {cache.dit_cache.autoregressive_index}"
        )
        cache.dit_cache.autoregressive_index = autoregressive_index
        clean_input = self.dit.generate(
            condition=WanDiTCondition(), cache=cache.dit_cache, rng=self.rng
        )

        if profile_events is not None:
            profile_events.toc_after_denoise.record()

        # 3. decode the clean input
        if hasattr(cache.detokenizer_cache, "autoregressive_index"):
            cache.detokenizer_cache.autoregressive_index = autoregressive_index
        decoded_video = self.detokenizer.decode(
            clean_input, cache=cache.detokenizer_cache
        )

        if profile_events is not None:
            profile_events.toc_after_decode.record()

        return decoded_video

    @torch.no_grad()
    def finalize(
        self,
        autoregressive_index: int,
        cache: Wan2_2PipelineCache,
    ) -> None:
        """
        Finalize the streaming inference. This will update the KV cache for the next block.
        """
        self.dit.finalize(cache.dit_cache, rng=self.rng)

        profile_events = cache.profile_events[-1]
        profile_events.toc_after_finalize.record()

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
