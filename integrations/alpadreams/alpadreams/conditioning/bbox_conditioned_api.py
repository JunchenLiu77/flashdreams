"""Wrapper combining a VideoModelAPI with HD map / bbox rendering for conditioning.

This module provides two main classes:
1. BboxConditionedT2V - Wraps a servable API with HD map rendering capabilities
2. BboxConditionedConformanceWrapper - Adds input validation

The gRPC server uses these with a session-based architecture where:
- The renderer is created during start_generation from scene_data
- The state includes the latent cache and renderer for subsequent calls
- Dynamic objects can be provided per-frame via object_info_per_frame
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic

import numpy as np
import torch
from alpadreams.conditioning.renderer import LudusRenderer
from alpadreams.conditioning.video_model_api import (
    BaseLatentCache,
    LatentCacheT,
    TextPrompt,
    VideoModelAPI,
    VideoModelConformanceWrapper,
)
from alpadreams.conditioning.world_scenario.data_types import SceneData
from alpadreams.conditioning.world_scenario.ftheta import FThetaCamera
from alpadreams.grpc.profiling_server import get_profiler, get_profiling_context
from torch import Tensor, nn


@dataclass
class BboxConditionedState(Generic[LatentCacheT]):
    """State for bbox-conditioned generation, including renderer."""

    latent_cache: (
        LatentCacheT | None
    )  # The model's latent cache (None if skip_video_generation)
    renderer: LudusRenderer


@dataclass
class GenerationOutput:
    """Output from start_generation or continue_generation."""

    state: BboxConditionedState  # Updated state for next call
    condition_frames: Tensor  # Rendered HDMap frames, shape [B, V, T, 3, H, W], uint8
    rgb_frames: Tensor | None = (
        None  # Generated video frames [B, V, T, 3, H, W] (None if skip_video_generation)
    )
    finalization_state: dict | None = None  # Finalization state from the video model


class BboxConditionedT2V(nn.Module, Generic[LatentCacheT]):
    """
    Combine a VideoModelAPI with an HD map / bbox renderer.

    This class exposes `start_generation` and `continue_generation` with a
    rendering-centric interface: instead of passing pre-built condition frames,
    the caller provides camera poses and frame IDs, and this class renders the
    condition frames internally before delegating to the underlying servable.

    For gRPC usage, the renderer is created during start_generation from scene_data,
    allowing session-based operation where static world data is loaded once.

    Note: This class does NOT inherit from VideoModelAPI because its method
    signatures differ (camera_poses + frame_timestamps_us instead of condition_frames).
    Access the underlying video model via `self.video_model_api` if needed.

    Multi-view vs single-view:
        Internally, all rendering is done with an explicit V (views) dimension.
        When the underlying ``VideoModelAPI`` is single-view (no ``n_cameras``
        attribute or ``n_cameras == 1``), the V dimension is squeezed before
        calling the model and re-inserted in the output so callers always see
        ``[B, V, T, 3, H, W]``.
    """

    def __init__(
        self,
        video_model: VideoModelAPI[LatentCacheT],
        wrap_with_conformance: bool = True,
        hdmap_color_version: str = "v3",
        bbox_color_version: str = "v3",
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """
        Args:
            video_model: The underlying video model API that performs video generation.
            wrap_with_conformance: If True and `video_model` is not already a
                `ConformanceWrapper`, wrap it to enable input/output validation.
            hdmap_color_version: HD map color scheme version (used when creating renderer).
            bbox_color_version: Bounding box color scheme version (used when creating renderer).
        """
        super().__init__()

        if wrap_with_conformance and not isinstance(
            video_model, VideoModelConformanceWrapper
        ):
            video_model = VideoModelConformanceWrapper(video_model)
        self.video_model_api = video_model
        self.hdmap_color_version = hdmap_color_version
        self.bbox_color_version = bbox_color_version
        self.device = device

        # Expose key attributes from the underlying servable for convenience
        self.latent_cache_type: type[BaseLatentCache] = video_model.latent_cache_type
        self.fps: int = video_model.fps
        self.frame_chunk_size: int = video_model.frame_chunk_size
        self.initial_frame_chunk_size: int = video_model.initial_frame_chunk_size
        self.video_resolution_wh: tuple[int, int] = video_model.video_resolution_wh

    @property
    def V_group(self) -> torch.distributed.ProcessGroup | None:
        """Process group for view dimension CP."""
        model = self.video_model_api
        if isinstance(model, VideoModelConformanceWrapper):
            model = model.model  # unwrap to get the real model
        return model.V_group

    @property
    def _is_multiview_model(self) -> bool:
        """Whether the underlying video model natively accepts multi-view (V>1) tensors.

        Single-view models expect ``[B, 3, H, W]`` / ``[B, T, 3, H, W]``.
        Multi-view models expect ``[B, V, 3, H, W]`` / ``[B, V, T, 3, H, W]``.
        """
        model = self.video_model_api
        if isinstance(model, VideoModelConformanceWrapper):
            model = model.model  # unwrap to get the real model
        return getattr(model, "n_cameras", 1) > 1

    @property
    def input_device(self) -> torch.device:
        """Device expected for input tensors (from the underlying video model)."""
        return self.video_model_api.input_device

    @property
    def output_device(self) -> torch.device:
        """Device of output tensors (from the underlying video model)."""
        return self.video_model_api.output_device

    def create_renderer(
        self,
        scene_data: SceneData,
        camera_names: list[str],
    ) -> LudusRenderer:
        """Create a renderer from scene data for one or more cameras.

        Args:
            scene_data: Static world / HD map data.
            camera_names: Camera names to include in the renderer.

        Returns:
            A renderer capable of rendering all listed cameras.
        """
        res_W, res_H = self.video_resolution_wh
        camera_models: dict[str, FThetaCamera] = {}

        for camera_name in camera_names:
            # Get or create camera model
            if scene_data.camera_models.get(camera_name) is None:
                # Create a default 120 FOV camera model.
                # from_numpy format: [cx, cy, width, height, *poly(6), is_bw_poly, linear_c, linear_d, linear_e]
                cx = res_W / 2
                cy = res_H / 2
                # For a 120 FOV equidistant camera: poly maps angle→pixel_dist
                # pixel_dist = f * angle, where f = width / (2 * FOV_half_rad)
                f = res_W / (2 * np.radians(60))  # 60 deg half-FOV
                intrinsics = np.array(
                    [cx, cy, res_W, res_H, f, 0, 0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0],
                    dtype=np.float64,
                )
                camera_model = FThetaCamera.from_numpy(intrinsics)
                scene_data.camera_models[camera_name] = camera_model

            camera_model_raw = scene_data.camera_models[camera_name]
            assert isinstance(camera_model_raw, FThetaCamera), (
                f"Currently only supporting FTheta cameras, got {type(camera_model_raw)=}"
            )
            camera_model = camera_model_raw

            # Resize camera model if needed
            if camera_model.height != res_H or camera_model.width != res_W:
                scale_h = res_H / camera_model.height
                scale_w = res_W / camera_model.width
                camera_model = FThetaCamera.from_numpy(camera_model.intrinsics.copy())
                camera_model.rescale(ratio_h=scale_h, ratio_w=scale_w)

            camera_models[camera_name] = camera_model

        return LudusRenderer(
            camera_models=camera_models,
            scene_data=scene_data,
            hdmap_color_version=self.hdmap_color_version,
            bbox_color_version=self.bbox_color_version,
            windowless=True,
            device=self.device,
        )

    def _render_condition_frames(
        self,
        renderer: LudusRenderer,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        object_info_per_frame: list[dict] | None = None,
    ) -> Tensor:
        """Render conditioning frames for all cameras.

        Args:
            renderer: The LudusRenderer to use.
            camera_names: Ordered list of camera names (defines V ordering).
            camera_poses_per_view: ``{camera_name: [T, 4, 4]}`` poses.
            frame_timestamps_us: Timestamps in microseconds for each frame.
            object_info_per_frame: Optional per-frame object info dicts.

        Returns:
            ``[B, V, T, 3, H, W]`` uint8 tensor on ``self.input_device`` (B=1).
        """
        # We use the same object info for all cameras
        obj_infos: list[dict | None] = []
        for i in range(len(frame_timestamps_us)):
            obj_info = None
            if object_info_per_frame is not None and i < len(object_info_per_frame):
                frame_obj_info = object_info_per_frame[i]
                if frame_obj_info:
                    obj_info = frame_obj_info
            obj_infos.append(obj_info)

        # NOTE: We do not support object info per frame for LudusRenderer yet
        for o in obj_infos:
            assert o is None, f"Object info not supported yet for LudusRenderer: {o}"

        # Render all frames and cameras in a single pass
        all_view_frames = renderer.render_all_frames_and_cameras(
            camera_names, camera_poses_per_view, frame_timestamps_us
        )
        assert all_view_frames.ndim == 5 and all_view_frames.dtype == torch.uint8

        # [1, V, T, 3, H, W]
        return all_view_frames.unsqueeze(0)

    def start_generation(
        self,
        text_prompts: list[TextPrompt],
        initial_rgb_frames: Tensor,
        renderer: LudusRenderer,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        skip_video_generation: bool = False,
    ) -> GenerationOutput:
        """Render initial condition frames and start video generation.

        Args:
            text_prompts: Text prompts for generation (length B, typically 1).
            initial_rgb_frames: Initial RGB images, shape ``[B, V, 3, H, W]``, uint8.
            renderer: Pre-created renderer for HDMap rendering.
            camera_names: Ordered list of camera names (defines V ordering).
            camera_poses_per_view: ``{camera_name: [T, 4, 4]}`` poses.
            frame_timestamps_us: Timestamps in microseconds for each frame.
            skip_video_generation: If True, only render HDMap without running the video model.

        Returns:
            GenerationOutput with ``condition_frames`` ``[B, V, T, 3, H, W]``
            and ``rgb_frames`` ``[B, V, T, 3, H, W]`` (or None).
        """
        # condition_frames: [B, V, T, 3, H, W]
        condition_frames = self._render_condition_frames(
            renderer, camera_names, camera_poses_per_view, frame_timestamps_us
        )

        if skip_video_generation:
            state: BboxConditionedState[LatentCacheT] = BboxConditionedState(
                latent_cache=None,
                renderer=renderer,
            )
            return GenerationOutput(
                state=state, condition_frames=condition_frames, rgb_frames=None
            )

        # Adapt shapes for the underlying video model:
        #   - Single-view models expect [B, 3, H, W] and [B, T, 3, H, W]
        #   - Multi-view models expect [B, V, 3, H, W] and [B, V, T, 3, H, W]
        if self._is_multiview_model:
            model_rgb = initial_rgb_frames  # [B, V, 3, H, W]
            model_cond = condition_frames  # [B, V, T, 3, H, W]
        else:
            model_rgb = initial_rgb_frames[:, 0]  # [B, 3, H, W]
            model_cond = condition_frames[:, 0]  # [B, T, 3, H, W]

        latent_cache, rgb_frames, finalization_state = (
            self.video_model_api.start_generation(
                text_prompts, model_rgb, model_cond, camera_names
            )
        )

        # Re-insert V dimension if model returned single-view output [B, T, 3, H, W]
        if not self._is_multiview_model:
            rgb_frames = rgb_frames.unsqueeze(1)  # → [B, 1, T, 3, H, W]

        state = BboxConditionedState(latent_cache=latent_cache, renderer=renderer)
        return GenerationOutput(
            state=state,
            condition_frames=condition_frames,
            rgb_frames=rgb_frames,
            finalization_state=finalization_state,
        )

    def continue_generation(
        self,
        state: BboxConditionedState,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        object_info_per_frame: list[dict] | None = None,
        skip_video_generation: bool = False,
        text_prompts: list[TextPrompt] | None = None,
    ) -> GenerationOutput:
        """Render condition frames and continue video generation.

        Args:
            state: State from previous generation (contains renderer and latent_cache).
            camera_names: Ordered list of camera names (defines V ordering).
            camera_poses_per_view: ``{camera_name: [T, 4, 4]}`` poses.
            frame_timestamps_us: Timestamps in microseconds for each frame.
            object_info_per_frame: Optional per-frame object info for dynamic actors.
            skip_video_generation: If True, only render HDMap without running the video model.
            text_prompts: Optional new text prompts.

        Returns:
            GenerationOutput with ``condition_frames`` ``[B, V, T, 3, H, W]``
            and ``rgb_frames`` ``[B, V, T, 3, H, W]`` (or None).
        """
        renderer = state.renderer

        profiler = get_profiler()
        session_id, chunk_idx = get_profiling_context()

        # condition_frames: [B, V, T, 3, H, W]
        with profiler.measure(
            "render_condition_frames", session_id=session_id, chunk_idx=chunk_idx
        ):
            condition_frames = self._render_condition_frames(
                renderer,
                camera_names,
                camera_poses_per_view,
                frame_timestamps_us,
                object_info_per_frame,
            )

        if skip_video_generation:
            return GenerationOutput(
                state=state, condition_frames=condition_frames, rgb_frames=None
            )

        if state.latent_cache is None:
            raise ValueError(
                "Cannot continue video generation: latent_cache is None "
                "(session was started with skip_video_generation=True)"
            )

        # Adapt shapes for single-view models (squeeze V=1)
        model_cond = (
            condition_frames if self._is_multiview_model else condition_frames[:, 0]
        )

        with profiler.measure(
            "video_model_api.continue_generation",
            session_id=session_id,
            chunk_idx=chunk_idx,
        ):
            new_latent_cache, rgb_frames, finalization_state = (
                self.video_model_api.continue_generation(
                    state.latent_cache, model_cond, text_prompts
                )
            )

        # Re-insert V dimension if model returned single-view output
        if not self._is_multiview_model:
            rgb_frames = rgb_frames.unsqueeze(1)  # → [B, 1, T, 3, H, W]

        new_state = BboxConditionedState(
            latent_cache=new_latent_cache, renderer=renderer
        )
        return GenerationOutput(
            state=new_state,
            condition_frames=condition_frames,
            rgb_frames=rgb_frames,
            finalization_state=finalization_state,
        )

    def cleanup(self, state: BboxConditionedState) -> None:
        """Clean up renderer resources."""
        if state.renderer is not None:
            state.renderer.cleanup()


class BboxConditionedConformanceWrapper(nn.Module, Generic[LatentCacheT]):
    """
    Conformance wrapper for BboxConditionedT2V that validates rendering-specific inputs.

    Only validates camera_poses and frame_timestamps_us. All other validation (text_prompts,
    rgb_frames, latent_cache, output shapes) is delegated to the underlying
    ConformanceWrapper on the servable API.
    """

    def __init__(self, api: BboxConditionedT2V[LatentCacheT]) -> None:
        super().__init__()
        self.api = api

        # Forward key attributes from the wrapped API
        self.latent_cache_type = api.latent_cache_type
        self.fps = api.fps
        self.frame_chunk_size = api.frame_chunk_size
        self.initial_frame_chunk_size = api.initial_frame_chunk_size
        self.video_resolution_wh = api.video_resolution_wh

    @property
    def V_group(self) -> torch.distributed.ProcessGroup | None:
        """Process group for view dimension CP."""
        return self.api.V_group

    @property
    def input_device(self) -> torch.device:
        return self.api.input_device

    @property
    def output_device(self) -> torch.device:
        return self.api.output_device

    def _validate_camera_poses(
        self, camera_poses: torch.Tensor, expected_length: int
    ) -> None:
        """Validate camera poses array."""
        assert isinstance(camera_poses, torch.Tensor), (
            f"camera_poses must be a torch.Tensor, got {type(camera_poses)}"
        )
        if not hasattr(camera_poses, "shape"):
            raise TypeError("camera_poses must be array-like with .shape")
        if len(camera_poses.shape) != 3 or camera_poses.shape[1:] != (4, 4):
            raise ValueError(
                f"camera_poses must be [num_frames, 4, 4], got {camera_poses.shape}"
            )
        if camera_poses.shape[0] != expected_length:
            raise ValueError(
                f"camera_poses has {camera_poses.shape[0]} frames, expected {expected_length}"
            )

    def _validate_frame_timestamps(
        self, frame_timestamps_us: list[int], expected_length: int
    ) -> None:
        """Validate frame_timestamps_us list."""
        if not isinstance(frame_timestamps_us, list):
            raise TypeError(
                f"frame_timestamps_us must be list, got {type(frame_timestamps_us).__name__}"
            )
        if not frame_timestamps_us:
            raise ValueError("frame_timestamps_us must not be empty")
        if not all(isinstance(t, int) for t in frame_timestamps_us):
            raise TypeError("All frame_timestamps_us must be integers")
        if len(frame_timestamps_us) != expected_length:
            raise ValueError(
                f"frame_timestamps_us length ({len(frame_timestamps_us)}) must be {expected_length}"
            )

    def create_renderer(
        self,
        scene_data: SceneData,
        camera_names: list[str],
    ) -> LudusRenderer:
        return self.api.create_renderer(scene_data, camera_names)

    def start_generation(
        self,
        text_prompts: list[TextPrompt],
        initial_rgb_frames: Tensor,
        renderer: LudusRenderer,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        skip_video_generation: bool = False,
    ) -> GenerationOutput:
        """Validate inputs and delegate to wrapped API."""
        for cam_name in camera_names:
            self._validate_camera_poses(
                camera_poses_per_view[cam_name], self.initial_frame_chunk_size
            )
        self._validate_frame_timestamps(
            frame_timestamps_us, self.initial_frame_chunk_size
        )

        return self.api.start_generation(
            text_prompts=text_prompts,
            initial_rgb_frames=initial_rgb_frames,
            renderer=renderer,
            camera_names=camera_names,
            camera_poses_per_view=camera_poses_per_view,
            frame_timestamps_us=frame_timestamps_us,
            skip_video_generation=skip_video_generation,
        )

    def continue_generation(
        self,
        state: BboxConditionedState,
        camera_names: list[str],
        camera_poses_per_view: dict[str, torch.Tensor],
        frame_timestamps_us: list[int],
        object_info_per_frame: list[dict] | None = None,
        skip_video_generation: bool = False,
        text_prompts: list[TextPrompt] | None = None,
    ) -> GenerationOutput:
        """Validate inputs and delegate to wrapped API."""
        for cam_name in camera_names:
            self._validate_camera_poses(
                camera_poses_per_view[cam_name], self.frame_chunk_size
            )
        self._validate_frame_timestamps(frame_timestamps_us, self.frame_chunk_size)

        return self.api.continue_generation(
            state=state,
            camera_names=camera_names,
            camera_poses_per_view=camera_poses_per_view,
            frame_timestamps_us=frame_timestamps_us,
            object_info_per_frame=object_info_per_frame,
            skip_video_generation=skip_video_generation,
            text_prompts=text_prompts,
        )

    def cleanup(self, state: BboxConditionedState) -> None:
        """Clean up renderer resources."""
        self.api.cleanup(state)
