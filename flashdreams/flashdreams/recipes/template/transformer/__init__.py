# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference :class:`Transformer` subclass for the template recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.infra.compile import compile_module
from flashdreams.infra.config import InstantiateConfig
from flashdreams.infra.cuda_graph import CUDAGraphWrapper
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
)
from flashdreams.infra.encoder import Encoder, NullEncoderConfig

from .network import TemplateDiT, TemplateDiTCache, TemplateDiTConfig


@dataclass(kw_only=True)
class TemplateTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for :class:`TemplateTransformer`.

    Cond and uncond branches own independent KV buffers: the residual
    stream diverges at the first context-bias addition.
    """

    network_cache: TemplateDiTCache
    """Conditional per-block KV cache and conditional context tokens."""

    network_cache_uncond: TemplateDiTCache | None = None
    """Unconditional cache. ``None`` disables CFG."""

    autoregressive_index: int = -1
    """AR step index for the chunk currently being processed; ``-1``
    before the first :meth:`start` call."""

    def start(self, autoregressive_index: int) -> None:
        """Snapshot the AR index and run per-block pre-update hooks.

        Hoisting ``before_update`` out of the network forward keeps the
        captured region shape-stable across AR steps.
        """
        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        """Run per-block post-update hooks."""
        self.network_cache.after_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.after_update(autoregressive_index)


@dataclass(kw_only=True)
class TemplateTransformerConfig(InstantiateConfig["TemplateTransformer"]):
    """Config for the template transformer.

    Bakes in the temporal layout (``len_t``, ``window_size_t``,
    ``sink_size_t``) and the CFG / compile knobs. Per-rollout spatial
    layout (``batch_size``, ``height``, ``width``) is supplied to
    :meth:`TemplateTransformer.initialize_autoregressive_cache` so one
    instance can serve multiple resolutions. CP size auto-detects from
    ``torch.distributed.get_world_size()`` at construction.
    """

    _target: type["TemplateTransformer"] = field(
        default_factory=lambda: TemplateTransformer
    )

    network: TemplateDiTConfig = field(default_factory=TemplateDiTConfig)
    """Underlying DiT network config."""

    context_encoder: InstantiateConfig[Any] = field(default_factory=NullEncoderConfig)
    """One-shot encoder applied to raw ``context`` inside
    :meth:`TemplateTransformer.initialize_autoregressive_cache`. The
    default :class:`~flashdreams.infra.encoder.NullEncoder` is identity;
    swap in a text or CLIP image encoder here."""

    dtype: torch.dtype = torch.bfloat16
    """Parameter and activation dtype."""

    checkpoint_path: str | None = None
    """Network checkpoint path for
    :func:`flashdreams.core.checkpoint.load.load_checkpoint`. ``None``
    keeps the random init."""

    len_t: int = 2
    """Pre-flatten latent frames per AR chunk."""

    window_size_t: int = 4
    """Pre-flatten sliding-window length, in temporal frames. The
    default keeps ``window_size_t == 2 * len_t`` so the streaming KV
    cache fills in exactly two AR steps before rolling; bidirectional
    variants must override with ``window_size_t == len_t``."""

    sink_size_t: int = 0
    """Pre-flatten sink length, in temporal frames."""

    guidance_scale: float = 1.0
    """CFG scale. ``1.0`` disables CFG; ``> 1.0`` requires a
    ``negative_context`` at cache build time."""

    compile_network: bool = False
    """Compile the network via :func:`flashdreams.infra.compile.compile_module`."""

    use_cuda_graph: bool = False
    """Wrap the network in :class:`CUDAGraphWrapper` for steady-state
    replay. The wrapper is built lazily inside
    :meth:`TemplateTransformer.initialize_autoregressive_cache` so each
    rollout gets static buffers sized to its ``(height, width)``."""

    cuda_graph_warmup_iters: int = 2
    """Eager calls before CUDA-graph capture; see :class:`CUDAGraphWrapper`."""

    @property
    def requires_negative_context_embeddings(self) -> bool:
        """``True`` when CFG is on (``guidance_scale > 1.0``)."""
        return self.guidance_scale > 1.0


class TemplateTransformer(Transformer[TemplateTransformerCache]):
    """Reference :class:`Transformer` subclass used by the template recipe."""

    config: TemplateTransformerConfig
    network: TemplateDiT
    context_encoder: Encoder

    def __init__(self, config: TemplateTransformerConfig) -> None:
        super().__init__(config)
        self.config = config

        if torch.distributed.is_initialized():
            self._cp_size = torch.distributed.get_world_size()
            self._cp_group = (
                torch.distributed.group.WORLD if self._cp_size > 1 else None
            )
        else:
            self._cp_size = 1
            self._cp_group = None

        self.network = TemplateDiT(config=config.network)
        self.network = self.network.to(dtype=config.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(cp_group=self._cp_group)

        if config.checkpoint_path is not None:
            state_dict = load_checkpoint(config.checkpoint_path)
            self.network.load_state_dict(state_dict)

        self.context_encoder = config.context_encoder.setup()

        if config.compile_network:
            self.network = compile_module(self.network)

        self._batch_size: int | None = None
        self._height: int | None = None
        self._width: int | None = None

        self._use_cuda_graph = config.use_cuda_graph
        self._network_call: CUDAGraphWrapper | None = None
        self._network_call_uncond: CUDAGraphWrapper | None = None

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank latent shape ``[batch_size, L/cp_size, in_channels]``.

        Populated by :meth:`initialize_autoregressive_cache`; reading
        it earlier asserts.
        """
        assert (
            self._batch_size is not None
            and self._height is not None
            and self._width is not None
        ), (
            "latent_shape requires an initialized rollout; call "
            "initialize_autoregressive_cache(..., height=..., width=...) "
            "first."
        )
        cfg = self.config
        L = cfg.len_t * self._height * self._width
        return (self._batch_size, L // self._cp_size, cfg.network.in_channels)

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        """Delegate to :meth:`TemplateDiT.patchify_and_maybe_split_cp`."""
        return self.network.patchify_and_maybe_split_cp(x)

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        """Delegate to :meth:`TemplateDiT.unpatchify_and_maybe_gather_cp`.

        Reads the per-rollout ``(height, width)``; asserts if called
        before :meth:`initialize_autoregressive_cache`.
        """
        assert self._height is not None and self._width is not None, (
            "unpatchify_and_maybe_gather_cp requires an initialized "
            "rollout; call initialize_autoregressive_cache(..., "
            "height=..., width=...) first."
        )
        return self.network.unpatchify_and_maybe_gather_cp(
            x, T=self.config.len_t, H=self._height, W=self._width
        )

    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        context: Tensor,
        negative_context: Tensor | None = None,
    ) -> TemplateTransformerCache:
        """Build a fully seeded cache for a new rollout.

        Runs ``context`` (and ``negative_context`` when CFG is on)
        through :attr:`context_encoder`, stashes the per-rollout
        ``(batch_size, height, width)``, and — when
        ``config.use_cuda_graph`` is set — builds fresh
        :class:`CUDAGraphWrapper` instances sized to this rollout.

        Args:
            height: Pre-patchify latent height.
            width: Pre-patchify latent width.
            context: Raw conditional context passed through
                :attr:`context_encoder`. The leading dim defines the
                rollout's ``batch_size``.
            negative_context: Raw unconditional context. Required iff
                ``config.guidance_scale > 1.0``.

        Returns:
            Seeded :class:`TemplateTransformerCache`.
        """
        cfg = self.config
        context_embeddings = self.context_encoder(input=context)
        batch_size, _, _ = context_embeddings.shape

        L_per_chunk = cfg.len_t * height * width
        assert L_per_chunk % self._cp_size == 0, (
            f"L = len_t*height*width ({L_per_chunk}) must be divisible by "
            f"cp_size ({self._cp_size})."
        )

        HW = height * width
        chunk_size = (cfg.len_t * HW) // self._cp_size
        window_size = (cfg.window_size_t * HW) // self._cp_size
        sink_size = (cfg.sink_size_t * HW) // self._cp_size

        device = context_embeddings.device
        dtype = cfg.dtype

        network_cache = self.network.initialize_cache(
            chunk_size=chunk_size,
            window_size=window_size,
            sink_size=sink_size,
            context=context_embeddings,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        network_cache_uncond: TemplateDiTCache | None = None
        if cfg.requires_negative_context_embeddings:
            assert negative_context is not None, (
                f"guidance_scale={cfg.guidance_scale} > 1.0 requires "
                "negative_context_embeddings."
            )
            negative_context_embeddings = self.context_encoder(input=negative_context)
            network_cache_uncond = self.network.initialize_cache(
                chunk_size=chunk_size,
                window_size=window_size,
                sink_size=sink_size,
                context=negative_context_embeddings,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        self._batch_size = batch_size
        self._height = height
        self._width = width

        # One wrapper per rollout: static buffers and the captured
        # graph are bound to this cache's KV pointers. The dispatch
        # threshold matches the KV cache's filling → steady transition
        # so the captured region only sees steady-state paths.
        if self._use_cuda_graph:
            self._cuda_graph_capture_ar_idx = (
                cfg.sink_size_t + cfg.window_size_t
            ) // cfg.len_t
            self._network_call = CUDAGraphWrapper(
                self.network, warmup_iters=cfg.cuda_graph_warmup_iters
            )
            self._network_call_uncond = CUDAGraphWrapper(
                self.network, warmup_iters=cfg.cuda_graph_warmup_iters
            )

        return TemplateTransformerCache(
            network_cache=network_cache,
            network_cache_uncond=network_cache_uncond,
        )

    def _select_network(self, cache: TemplateTransformerCache, *, uncond: bool) -> Any:
        # Filling phase: eager ``.drain`` (drains Inductor autotune and
        # exercises the KV cache's slice-returning filling path).
        # Steady phase: ``wrapper.__call__`` (warmup + capture + replay).
        # Capturing in the filling phase would bake slice pointers into
        # the graph and read stale storage after the cache rolls.
        if not self._use_cuda_graph:
            return self.network
        network_call = self._network_call_uncond if uncond else self._network_call
        assert isinstance(network_call, CUDAGraphWrapper), (
            "predict_flow called before initialize_autoregressive_cache "
            "while use_cuda_graph=True."
        )
        return (
            network_call.drain
            if cache.autoregressive_index < self._cuda_graph_capture_ar_idx
            else network_call
        )

    def _predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: TemplateTransformerCache,
        network_cache: TemplateDiTCache,
        control: Tensor | None,
        *,
        uncond: bool,
    ) -> Tensor:
        assert cache.autoregressive_index >= 0, (
            "Cache.start(autoregressive_index) must be called before "
            "predict_flow (DiffusionModel.generate handles this)."
        )
        return self._select_network(cache, uncond=uncond)(
            noisy_latent,
            timesteps=timestep,
            cache=network_cache,
            control=control,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: TemplateTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        """Run the cond branch and merge in the uncond branch when CFG is on.

        Args:
            noisy_latent: ``[B, L/cp, in_channels]`` per-rank noisy latent.
            timestep: Scalar timestep.
            cache: Per-rollout cache; ``cache.autoregressive_index`` must
                be set by a prior :meth:`TemplateTransformerCache.start`.
            input: Encoded control latent, or ``None`` to skip the
                per-token control bias.

        Returns:
            ``[B, L/cp, in_channels]`` flow prediction with CFG applied
            when ``cache.network_cache_uncond`` is populated.
        """
        flow_cond = self._predict_flow(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            network_cache=cache.network_cache,
            control=input,
            uncond=False,
        )
        if cache.network_cache_uncond is None:
            return flow_cond
        else:
            flow_uncond = self._predict_flow(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                network_cache=cache.network_cache_uncond,
                control=input,
                uncond=True,
            )
            return flow_uncond + self.config.guidance_scale * (flow_cond - flow_uncond)
