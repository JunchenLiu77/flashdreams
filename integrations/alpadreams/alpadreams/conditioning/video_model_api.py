"""Defines the base API for video-to-video generation models such that they can be served via gRPC."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Type, TypeVar

import torch
from torch import Tensor, nn


class BaseLatentCache(nn.Module, ABC):
    """
    Base class for latent caches (e.g. KV cache) for video generation.

    It is not intended as a learnable `nn.Module`, we simply reuse it for convenience features like `.to(device)` and `.device`.
    """

    batch_size: int
    """ The batch size of the latent cache. """

    history_length_in_frames: int
    """ The duration of video already generated with this latent cache. """

    @property
    def device(self) -> torch.device:
        """The device on which the latent cache is stored."""
        return next(self.parameters()).device


LatentCacheT = TypeVar("LatentCacheT", bound=BaseLatentCache)
""" Generic type representing the type used by the servable model as its latent cache. """


@dataclass
class TextPrompt:
    """A text prompt for video generation (single video)."""

    positive: str
    negative: str | None = None

    # precomputed embeddings
    positive_embeddings: Tensor | None = None
    negative_embeddings: Tensor | None = None


class VideoModelAPI(nn.Module, ABC, Generic[LatentCacheT]):
    """
    Base class for video models compatible with the gRPC server.

    Subclass this class and implement the abstract methods to create a video model which works with the gRPC server.
    """

    latent_cache_type: Type[LatentCacheT]
    """ The type of the latent cache used by this video model. Must be a subclass of `BaseLatentCache`. """

    fps: int
    """ The frames per second for generated videos. """

    frame_chunk_size: int
    """ The frame chunk size supported by this class. If many are supported, return an example and override is_supported_frame_chunk_size. """

    initial_frame_chunk_size: int
    """ The frame chunk size for the initial block. May differ from frame_chunk_size for subsequent blocks. If many are supported, return an example and override is_supported_initial_frame_chunk_size. """

    def is_supported_frame_chunk_size(self, frame_chunk_size: int) -> bool:
        """Checks if the given frame chunk size is supported by this class."""
        return frame_chunk_size == self.frame_chunk_size

    def is_supported_initial_frame_chunk_size(
        self, initial_frame_chunk_size: int
    ) -> bool:
        """Checks if the given initial frame chunk size is supported by this class."""
        return initial_frame_chunk_size == self.initial_frame_chunk_size

    video_resolution_wh: tuple[int, int]
    """ The video resolution supported by this class. If many are supported, return an example and override is_supported_resolution. """

    def is_supported_resolution(self, resolution_wh: tuple[int, int]) -> bool:
        """Checks if the given resolution is supported by this class."""
        return resolution_wh == self.video_resolution_wh

    @property
    def input_device(self) -> torch.device:
        """The device on which the input tensors are expected to be on. By default it is the device of the first parameter."""
        return next(self.parameters()).device

    @property
    def output_device(self) -> torch.device:
        """The device on which the output tensors are expected to be on. By default it is the same as the input device."""
        return self.input_device

    @abstractmethod
    def start_generation(
        self,
        text_prompts: list[TextPrompt],
        initial_rgb_frames: Tensor,
        initial_condition_frames: Tensor,
        view_names: list[str],
    ) -> tuple[LatentCacheT, Tensor, dict]:
        """
        Starts the generation of a batch of B new videos.

        Args:
            text_prompts: The text prompts to use for the generation, length is B.
            initial_rgb_frames: The initial RGB frames to use for the generation. uint8, on `self.input_device`, shape [B, 3, H, W].
            initial_condition_frames: The initial condition frames to use for the generation. uint8, on `self.input_device`, shape [B, T, 3, H, W].
            view_names: The names of the views to use for the generation.

        Returns:
            - latent_cache: The latent cache of type `self.latent_cache_type`.
            - output_frames: The output frames. uint8, on `self.output_device`, shape [B, T, 3, H, W].
        """
        ...

    @abstractmethod
    def continue_generation(
        self,
        latent_cache: LatentCacheT,
        condition_frames: Tensor,
        text_prompts: Optional[list[TextPrompt]] = None,
    ) -> tuple[LatentCacheT, Tensor, dict]:
        """
        Continues the generation of a batch of B videos with new condition frames and (optionally) new text prompts.

        A model MUST be capable of generating without new text prompts, it MAY ignore them if it doesn't support re-prompting.

        Args:
            latent_cache: The latent cache of type `self.latent_cache_type`. Must match `latent_cache.batch_size == B`.
            condition_frames: The condition frames to use for the generation. uint8, on `self.input_device`, shape [B, T, 3, H, W].
            text_prompts: The text prompts to use for the generation, length matches `latent_cache.batch_size`.

        Returns:
            - latent_cache: The latent cache of type `self.latent_cache_type`.
            - output_frames: The output frames. uint8, on `self.output_device`, shape [B, T, 3, H, W].
        """
        ...


class VideoModelConformanceWrapper(VideoModelAPI[LatentCacheT], Generic[LatentCacheT]):
    """Wraps a model implementing the VideoModelAPI interface and adds sanity checks on input/output to catch errors early."""

    def __init__(self, model: VideoModelAPI[LatentCacheT]):
        super().__init__()
        self.model = model

        # Forward class attributes from the wrapped model
        self.latent_cache_type = model.latent_cache_type
        self.fps = model.fps
        self.frame_chunk_size = model.frame_chunk_size
        self.initial_frame_chunk_size = model.initial_frame_chunk_size
        self.video_resolution_wh = model.video_resolution_wh

    def is_supported_frame_chunk_size(self, frame_chunk_size: int) -> bool:
        return self.model.is_supported_frame_chunk_size(frame_chunk_size)

    def is_supported_initial_frame_chunk_size(
        self, initial_frame_chunk_size: int
    ) -> bool:
        return self.model.is_supported_initial_frame_chunk_size(
            initial_frame_chunk_size
        )

    def is_supported_resolution(self, resolution_wh: tuple[int, int]) -> bool:
        return self.model.is_supported_resolution(resolution_wh)

    @property
    def input_device(self) -> torch.device:
        return self.model.input_device

    @property
    def output_device(self) -> torch.device:
        return self.model.output_device

    def start_generation(
        self,
        text_prompts: list[TextPrompt],
        initial_rgb_frames: Tensor,
        initial_condition_frames: Tensor,
    ) -> tuple[LatentCacheT, Tensor, dict]:
        """Validate inputs, call wrapped model, validate outputs."""
        # Validate text_prompts
        assert isinstance(text_prompts, list), (
            f"text_prompts must be a list, got {type(text_prompts)}"
        )
        assert len(text_prompts) > 0, "text_prompts must not be empty"
        assert all(isinstance(p, TextPrompt) for p in text_prompts), (
            "All items in text_prompts must be TextPrompt instances"
        )
        B = len(text_prompts)

        # Validate initial_rgb_frames
        assert isinstance(initial_rgb_frames, Tensor), (
            f"initial_rgb_frames must be a Tensor, got {type(initial_rgb_frames)}"
        )
        assert initial_rgb_frames.dtype == torch.uint8, (
            f"initial_rgb_frames must be uint8, got {initial_rgb_frames.dtype}"
        )
        assert initial_rgb_frames.device == self.input_device, (
            f"initial_rgb_frames must be on {self.input_device}, got {initial_rgb_frames.device}"
        )
        assert initial_rgb_frames.ndim == 4, (
            f"initial_rgb_frames must have 4 dimensions [B, 3, H, W], got {initial_rgb_frames.ndim}"
        )
        assert initial_rgb_frames.shape[0] == B, (
            f"initial_rgb_frames batch size {initial_rgb_frames.shape[0]} must match text_prompts length {B}"
        )
        assert initial_rgb_frames.shape[1] == 3, (
            f"initial_rgb_frames must have 3 channels, got {initial_rgb_frames.shape[1]}"
        )
        H, W = initial_rgb_frames.shape[2], initial_rgb_frames.shape[3]
        assert self.is_supported_resolution((W, H)), (
            f"Resolution {(W, H)} not supported. Supported: {self.video_resolution_wh}"
        )

        # Validate initial_condition_frames
        assert isinstance(initial_condition_frames, Tensor), (
            f"initial_condition_frames must be a Tensor, got {type(initial_condition_frames)}"
        )
        assert initial_condition_frames.dtype == torch.uint8, (
            f"initial_condition_frames must be uint8, got {initial_condition_frames.dtype}"
        )
        assert initial_condition_frames.device == self.input_device, (
            f"initial_condition_frames must be on {self.input_device}, got {initial_condition_frames.device}"
        )
        assert initial_condition_frames.ndim == 5, (
            f"initial_condition_frames must have 5 dimensions [B, T, 3, H, W], got {initial_condition_frames.ndim}"
        )
        assert initial_condition_frames.shape[0] == B, (
            f"initial_condition_frames batch size {initial_condition_frames.shape[0]} must match text_prompts length {B}"
        )
        T_init = initial_condition_frames.shape[1]
        assert self.is_supported_frame_chunk_size(T_init), (
            f"Frame chunk size {T_init} not supported. Supported: {self.frame_chunk_size}"
        )
        assert initial_condition_frames.shape[2] == 3, (
            f"initial_condition_frames must have 3 channels, got {initial_condition_frames.shape[2]}"
        )
        assert (
            initial_condition_frames.shape[3] == H
            and initial_condition_frames.shape[4] == W
        ), (
            f"initial_condition_frames resolution [{initial_condition_frames.shape[3]}, {initial_condition_frames.shape[4]}] "
            f"must match initial_rgb_frames resolution [{H}, {W}]"
        )

        # Call wrapped model
        latent_cache, output_frames, finalization_state = self.model.start_generation(
            text_prompts, initial_rgb_frames, initial_condition_frames
        )

        # Validate latent_cache
        assert isinstance(latent_cache, self.latent_cache_type), (
            f"latent_cache must be of type {self.latent_cache_type}, got {type(latent_cache)}"
        )
        assert latent_cache.batch_size == B, (
            f"latent_cache batch_size {latent_cache.batch_size} must match input batch size {B}"
        )
        assert latent_cache.history_length_in_frames >= 0, (
            f"latent_cache.history_length_in_frames must be non-negative, got {latent_cache.history_length_in_frames}"
        )

        # Validate output_frames
        assert isinstance(output_frames, Tensor), (
            f"output_frames must be a Tensor, got {type(output_frames)}"
        )
        assert output_frames.dtype == torch.uint8, (
            f"output_frames must be uint8, got {output_frames.dtype}"
        )
        assert output_frames.device == self.output_device, (
            f"output_frames must be on {self.output_device}, got {output_frames.device}"
        )
        assert output_frames.ndim == 5, (
            f"output_frames must have 5 dimensions [B, T, 3, H, W], got {output_frames.ndim}"
        )
        assert output_frames.shape[0] == B, (
            f"output_frames batch size {output_frames.shape[0]} must match input batch size {B}"
        )
        T_out = output_frames.shape[1]
        assert self.is_supported_frame_chunk_size(T_out), (
            f"Output frame chunk size {T_out} not supported. Supported: {self.frame_chunk_size}"
        )
        assert output_frames.shape[2] == 3, (
            f"output_frames must have 3 channels, got {output_frames.shape[2]}"
        )
        assert output_frames.shape[3] == H and output_frames.shape[4] == W, (
            f"output_frames resolution [{output_frames.shape[3]}, {output_frames.shape[4]}] "
            f"must match input resolution [{H}, {W}]"
        )

        return latent_cache, output_frames, finalization_state

    def continue_generation(
        self,
        latent_cache: LatentCacheT,
        condition_frames: Tensor,
        text_prompts: Optional[list[TextPrompt]] = None,
    ) -> tuple[LatentCacheT, Tensor, dict]:
        """Validate inputs, call wrapped model, validate outputs."""
        # Validate latent_cache
        assert isinstance(latent_cache, self.latent_cache_type), (
            f"latent_cache must be of type {self.latent_cache_type}, got {type(latent_cache)}"
        )
        B = latent_cache.batch_size
        assert B > 0, f"latent_cache.batch_size must be positive, got {B}"

        # Validate condition_frames
        assert isinstance(condition_frames, Tensor), (
            f"condition_frames must be a Tensor, got {type(condition_frames)}"
        )
        assert condition_frames.dtype == torch.uint8, (
            f"condition_frames must be uint8, got {condition_frames.dtype}"
        )
        assert condition_frames.device == self.input_device, (
            f"condition_frames must be on {self.input_device}, got {condition_frames.device}"
        )
        assert condition_frames.ndim == 5, (
            f"condition_frames must have 5 dimensions [B, T, 3, H, W], got {condition_frames.ndim}"
        )
        assert condition_frames.shape[0] == B, (
            f"condition_frames batch size {condition_frames.shape[0]} must match latent_cache.batch_size {B}"
        )
        T_cond = condition_frames.shape[1]
        assert self.is_supported_frame_chunk_size(T_cond), (
            f"Frame chunk size {T_cond} not supported. Supported: {self.frame_chunk_size}"
        )
        assert condition_frames.shape[2] == 3, (
            f"condition_frames must have 3 channels, got {condition_frames.shape[2]}"
        )
        H, W = condition_frames.shape[3], condition_frames.shape[4]
        assert self.is_supported_resolution((W, H)), (
            f"Resolution {(W, H)} not supported. Supported: {self.video_resolution_wh}"
        )

        # Validate text_prompts if provided
        if text_prompts is not None:
            assert isinstance(text_prompts, list), (
                f"text_prompts must be a list, got {type(text_prompts)}"
            )
            assert len(text_prompts) == B, (
                f"text_prompts length {len(text_prompts)} must match latent_cache.batch_size {B}"
            )
            assert all(isinstance(p, TextPrompt) for p in text_prompts), (
                "All items in text_prompts must be TextPrompt instances"
            )

        # Store initial history length for validation
        initial_history_length = latent_cache.history_length_in_frames

        # Call wrapped model
        updated_cache, output_frames, finalization_state = (
            self.model.continue_generation(latent_cache, condition_frames, text_prompts)
        )

        # Validate updated_cache
        assert isinstance(updated_cache, self.latent_cache_type), (
            f"updated_cache must be of type {self.latent_cache_type}, got {type(updated_cache)}"
        )
        assert updated_cache.batch_size == B, (
            f"updated_cache batch_size {updated_cache.batch_size} must match input batch size {B}"
        )
        assert updated_cache.history_length_in_frames >= initial_history_length, (
            f"updated_cache.history_length_in_frames {updated_cache.history_length_in_frames} "
            f"must not decrease from initial {initial_history_length}"
        )

        # Validate output_frames
        assert isinstance(output_frames, Tensor), (
            f"output_frames must be a Tensor, got {type(output_frames)}"
        )
        assert output_frames.dtype == torch.uint8, (
            f"output_frames must be uint8, got {output_frames.dtype}"
        )
        assert output_frames.device == self.output_device, (
            f"output_frames must be on {self.output_device}, got {output_frames.device}"
        )
        assert output_frames.ndim == 5, (
            f"output_frames must have 5 dimensions [B, T, 3, H, W], got {output_frames.ndim}"
        )
        assert output_frames.shape[0] == B, (
            f"output_frames batch size {output_frames.shape[0]} must match input batch size {B}"
        )
        T_out = output_frames.shape[1]
        assert self.is_supported_frame_chunk_size(T_out), (
            f"Output frame chunk size {T_out} not supported. Supported: {self.frame_chunk_size}"
        )
        assert output_frames.shape[2] == 3, (
            f"output_frames must have 3 channels, got {output_frames.shape[2]}"
        )
        assert output_frames.shape[3] == H and output_frames.shape[4] == W, (
            f"output_frames resolution [{output_frames.shape[3]}, {output_frames.shape[4]}] "
            f"must match input resolution [{H}, {W}]"
        )

        return updated_cache, output_frames, finalization_state


# Default text prompt for demos (generic)
DEFAULT_POSITIVE_PROMPT = (
    "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in "
    "the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon "
    "light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning "
    "capturing the large furry mammal with beautiful photography, depth of field."
)
# Autonomous driving prompt
AV_POSITIVE_PROMPT = (
    "Driving scene from a front-facing car camera. Urban environment with roads, vehicles, pedestrians, "
    "traffic signs, and buildings. Clear visibility, realistic lighting, photorealistic quality. "
    "High resolution dashcam footage of city driving."
)


def get_av_text_prompts(
    batch_size: int = 1,
    device: torch.device | str = "cuda",
    text_prompt: str = AV_POSITIVE_PROMPT,
) -> list[TextPrompt]:
    """Get autonomous driving text prompts."""
    del device
    return [TextPrompt(positive=text_prompt)] * batch_size
