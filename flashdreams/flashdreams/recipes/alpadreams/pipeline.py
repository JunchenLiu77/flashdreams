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

"""Alpadreams streaming inference pipeline (Cosmos DiT + I2V + HDMap).

Project-level :class:`StreamInferencePipeline` subclass (no wrapper
indirection) for the alpadreams Cosmos DiT checkpoints. The pipeline
wires:

- :class:`CosmosReason1TextEncoder` — one-shot at rollout start.
- A first-frame Wan VAE encoder (project-level ``image_encoder``) —
  one-shot at rollout start; its output seeds the long-lived AR cache
  (mask injection at AR step 0 only).
- The infra :attr:`StreamInferencePipelineConfig.encoder` —
  per-AR-step HDMap encoder (Wan VAE *or* PixelShuffle pseudo-VAE),
  required.
- The infra :attr:`StreamInferencePipelineConfig.decoder` —
  per-AR-step video decoder (Wan VAE *or* TAEHV).
- A :class:`DiffusionModelConfig` over :class:`CosmosTransformerConfig`.

I2V is implemented by :class:`CosmosTransformer` itself (mask injection
into ``noisy_latent`` and ``postprocess_clean_latent``); the per-AR-step
``input`` is the encoded HDMap chunk routed through the infra
encoder slot.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TypeAlias

import torch
from torch import Tensor

from flashdreams.core.distributed.context_parallel import (
    cat_outputs_cp,
    split_inputs_cp,
    split_inputs_cp_object_list,
)
from flashdreams.infra.decoder import DecoderAutoregressiveCache
from flashdreams.infra.encoder import EncoderAutoregressiveCache
from flashdreams.infra.encoder.text.cosmos_qwen import (
    CosmosReason1TextEncoder,
    CosmosReason1TextEncoderConfig,
)
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineCache,
    StreamInferencePipelineConfig,
)
from flashdreams.recipes.alpadreams.transformer import (
    CosmosTransformerCache,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    WanVAEEncoder,
    WanVAEEncoderConfig,
)

AlpadreamsPipelineCache: TypeAlias = StreamInferencePipelineCache[
    EncoderAutoregressiveCache,  # EncCacheT
    CosmosTransformerCache,  # TransformerCacheT
    DecoderAutoregressiveCache,  # DecCacheT
]


@dataclass(kw_only=True)
class AlpadreamsPipelineConfig(StreamInferencePipelineConfig):
    """Hyperparameters for :class:`AlpadreamsPipeline`.

    Extends :class:`StreamInferencePipelineConfig` (inherits
    ``diffusion_model``, ``encoder``, ``decoder``) with the project-level
    text and first-frame image encoders. ``encoder`` MUST be set (the
    per-AR-step HDMap encoder); ``diffusion_model.transformer`` must be
    a :class:`CosmosTransformerConfig`.
    """

    _target: type["StreamInferencePipeline"] = field(
        default_factory=lambda: AlpadreamsPipeline
    )  # ty:ignore[invalid-assignment]

    text_encoder: CosmosReason1TextEncoderConfig | None = field(
        default_factory=CosmosReason1TextEncoderConfig
    )
    """Cosmos-Reason1 text encoder, run once at the start of each rollout.

    Set to ``None`` to skip loading the encoder entirely; in that case
    rollouts must be initialized via
    :meth:`AlpadreamsPipeline.initialize_cache_from_embeddings` with
    embeddings precomputed elsewhere (e.g. by
    :meth:`AlpadreamsPipeline.precompute_embeddings`)."""

    image_encoder: WanVAEEncoderConfig | None = field(
        default_factory=WanVAEEncoderConfig
    )
    """One-shot Wan VAE encoder used to encode the first-frame image
    (per-rollout, NOT per-AR-step). Pin its checkpoint to the same VAE
    that produced the network's training distribution. Set to ``None``
    to skip loading; see ``text_encoder`` above."""


class AlpadreamsPipeline(
    StreamInferencePipeline[
        EncoderAutoregressiveCache,  # EncCacheT
        CosmosTransformerCache,  # TransformerCacheT
        DecoderAutoregressiveCache,  # DecCacheT
    ]
):
    """Streaming alpadreams inference pipeline (Cosmos DiT + HDMap + I2V mask).

    Usage::

        config = ALPADREAMS_CONFIG_BUILDERS["sv_..."](device=device)
        pipeline = config.setup().to(device=device)

        cache = pipeline.initialize_cache(
            text=[["A driving scene..."]],  # [B, V] (V=1 for single-view)
            image=first_frames,             # [B, V, 1, 3, H, W]
            view_names=["camera_front_..."],
        )
        for i in range(num_blocks):
            chunk = pipeline.generate(i, cache, hdmap=hdmap_chunk_i)
            pipeline.finalize(i, cache)
    """

    text_encoder: CosmosReason1TextEncoder | None
    image_encoder: WanVAEEncoder | None

    def __init__(self, config: AlpadreamsPipelineConfig) -> None:
        super().__init__(config)
        self.text_encoder = (  # ty:ignore[invalid-assignment]
            config.text_encoder.setup() if config.text_encoder is not None else None
        )
        self.image_encoder = (  # ty:ignore[invalid-assignment]
            config.image_encoder.setup() if config.image_encoder is not None else None
        )

        assert self.encoder is not None, (
            "AlpadreamsPipeline requires a per-AR-step HDMap encoder; "
            "set StreamInferencePipelineConfig.encoder."
        )

        transformer = self.diffusion_model.transformer
        self._len_t_latent: int = transformer.config.len_t  # ty:ignore[unresolved-attribute]
        decoder = self.decoder
        assert decoder is not None and hasattr(decoder, "TEMPORAL_COMPRESSION_RATIO"), (
            f"Decoder {type(decoder).__name__} must expose "
            "TEMPORAL_COMPRESSION_RATIO (e.g. WanVAEDecoder, TeahvVAEDecoder)."
        )
        self._decoder_temporal_compression: int = decoder.TEMPORAL_COMPRESSION_RATIO  # ty:ignore[invalid-assignment]

        # Take the view split outside of the transformer, so that VAE does not do duplicated job.
        self.V_group = transformer.cp_groups.V_group  # ty:ignore[unresolved-attribute]
        self.V_size = transformer.cp_groups.V_size  # ty:ignore[unresolved-attribute]
        transformer.cp_groups.V_group = None  # ty:ignore[invalid-assignment]
        transformer.config.num_views //= self.V_size  # ty:ignore[unresolved-attribute]

    @property
    def device(self) -> torch.device:
        return self.diffusion_model.device

    # ------------------------------------------------------------------
    # AR rollout API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def initialize_cache(  # type: ignore[override]
        self,
        text: list[list[str]],
        image: Tensor,
        view_names: list[str] | None = None,
    ) -> AlpadreamsPipelineCache:
        """Initialize the per-rollout cache.

        Args:
            text: ``[B, V]`` nested list of prompts (one per view).
            image: First frame pixel tensor of shape ``[B, V, 1, 3, H, W]``
                in ``[-1, 1]``. ``H`` / ``W`` must match
                ``transformer.config.height/width *
                WanVAEDecoder.SPATIAL_COMPRESSION_RATIO``.
            view_names: List of view names (length ``V``); required when
                ``num_views > 1``.
        """
        assert self.text_encoder is not None and self.image_encoder is not None, (
            "initialize_cache(text=, image=) requires text_encoder and "
            "image_encoder to be loaded. They are None either because the "
            "configs were set to None at construction or because "
            "release_oneshot_encoders() has been called. If you have "
            "precomputed embeddings, use initialize_cache_from_embeddings()."
        )
        assert isinstance(text, list) and len(text) > 0 and isinstance(text[0], list), (
            f"text must be a [B, V] nested list of prompts, got {type(text)}"
        )

        text_embeddings = torch.stack(
            [self.text_encoder(t) for t in text], dim=0
        )  # [B, V, L, D]
        image_embeddings = self.image_encoder(image)  # [B, V, 1, Cl, Hl, Wl]

        return self.initialize_cache_from_embeddings(
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            view_names=view_names,
        )

    @torch.no_grad()
    def initialize_cache_from_embeddings(
        self,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
        view_names: list[str] | None = None,
    ) -> AlpadreamsPipelineCache:
        """Initialize the per-rollout cache from precomputed embeddings.

        Use this when ``text_encoder`` / ``image_encoder`` are not loaded
        (e.g. the pipeline was built with both configs set to ``None`` to
        save VRAM, and embeddings were computed offline by
        :meth:`precompute_embeddings`).

        Args:
            text_embeddings: ``[B, V, L, D]`` tensor as produced by the
                Cosmos-Reason1 text encoder. Will be moved to
                ``self.device`` if needed. NOT yet CP-split: pass the
                full multi-view tensor; the split is applied here.
            image_embeddings: ``[B, V, 1, Cl, Hl, Wl]`` tensor as
                produced by the Wan VAE first-frame encoder. Same
                device / split contract as ``text_embeddings``.
            view_names: List of view names (length ``V``); required when
                ``num_views > 1``.
        """
        text_embeddings = text_embeddings.to(device=self.device)
        image_embeddings = image_embeddings.to(device=self.device)

        # distribute multi-view
        text_embeddings = split_inputs_cp(
            text_embeddings,
            seq_dim=1,
            cp_group=self.V_group,  # ty:ignore[invalid-argument-type]
        )
        image_embeddings = split_inputs_cp(
            image_embeddings,
            seq_dim=1,
            cp_group=self.V_group,  # ty:ignore[invalid-argument-type]
        )
        view_names = split_inputs_cp_object_list(view_names, cp_group=self.V_group)  # ty:ignore[invalid-argument-type]

        return super().initialize_cache(
            transformer_context={
                "text_embeddings": text_embeddings,
                "image_embeddings": image_embeddings,
                "view_names": view_names,
            },
        )

    @torch.no_grad()
    def precompute_embeddings(
        self,
        text: list[list[str]],
        image: Tensor,
    ) -> dict[str, Tensor]:
        """Run only the one-shot encoders and return their outputs on CPU.

        Pair with :meth:`initialize_cache_from_embeddings`: ``torch.save``
        the returned dict, then in a separate process build a pipeline
        with ``text_encoder=None`` / ``image_encoder=None`` and rebuild
        the cache from the loaded tensors. The returned tensors are NOT
        CP-split; the split happens inside
        :meth:`initialize_cache_from_embeddings` at load time, so the
        same precomputed file works for any CP world size.

        Args:
            text: ``[B, V]`` nested list of prompts (one per view).
            image: ``[B, V, 1, 3, H, W]`` first-frame pixel tensor in
                ``[-1, 1]``.

        Returns:
            ``{"text_embeddings": [B, V, L, D],
            "image_embeddings": [B, V, 1, Cl, Hl, Wl]}`` on CPU.
        """
        assert self.text_encoder is not None and self.image_encoder is not None, (
            "precompute_embeddings requires text_encoder and image_encoder "
            "to be loaded; build the pipeline with both configs non-None."
        )
        assert isinstance(text, list) and len(text) > 0 and isinstance(text[0], list), (
            f"text must be a [B, V] nested list of prompts, got {type(text)}"
        )
        text_embeddings = torch.stack(
            [self.text_encoder(t) for t in text], dim=0
        )  # [B, V, L, D]
        image_embeddings = self.image_encoder(image)  # [B, V, 1, Cl, Hl, Wl]
        return {
            "text_embeddings": text_embeddings.cpu(),
            "image_embeddings": image_embeddings.cpu(),
        }

    def release_oneshot_encoders(self) -> None:
        """Free the per-rollout text and first-frame image encoders.

        Both encoders are only needed inside :meth:`initialize_cache`; the
        AR loop reads their outputs from ``cache.transformer_context``.
        Cosmos-Reason1-7B alone is ~14 GB in bf16, so dropping it after
        ``initialize_cache`` reclaims significant VRAM for the AR rollout.

        Idempotent. After calling this, :meth:`initialize_cache` will
        raise a clear assertion (the encoders are now ``None``); only
        call it from contexts that run a single rollout per pipeline
        instance (e.g. one-shot demos). Long-lived hosts that reuse the
        pipeline across sessions (e.g. the gRPC server) must not call it.
        """
        # Set to None (not delattr) so that ``initialize_cache``'s
        # existing ``is not None`` guard fires with a useful message
        # instead of an AttributeError.
        self.text_encoder = None
        self.image_encoder = None
        # nn.Module instances commonly form reference cycles (parent <->
        # child, hooks, etc.) that the refcount path alone won't break,
        # so force a GC pass before asking the allocator to release the
        # freed CUDA blocks.
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        autoregressive_index: int,
        cache: AlpadreamsPipelineCache,
        hdmap: Tensor,
    ) -> Tensor:
        """Generate one decoded video chunk.

        Args:
            autoregressive_index: AR step index (``0``-based).
            cache: The per-rollout pipeline cache.
            hdmap: Per-AR-step HDMap pixel tensor of shape
                ``[B, V, T, 3, H, W]`` in ``[-1, 1]``. ``T`` must equal
                :meth:`get_num_frames(autoregressive_index)`.

        Returns:
            Decoded video chunk of shape ``[B, V, T, 3, H, W]`` in
            ``[-1, 1]``.
        """
        # distribute multi-view
        hdmap = split_inputs_cp(hdmap, seq_dim=1, cp_group=self.V_group)  # ty:ignore[invalid-argument-type]

        # generate
        output = super().generate(
            autoregressive_index=autoregressive_index,
            cache=cache,
            input=hdmap,
        )

        # gather multi-view
        output = cat_outputs_cp(output, seq_dim=1, cp_group=self.V_group)  # ty:ignore[invalid-argument-type]
        return output

    def get_num_frames(self, autoregressive_index: int) -> int:
        """Number of decoded video frames produced by AR step ``autoregressive_index``.

        AR step 0 emits the streaming-VAE anchor frame plus
        ``(len_t - 1) * temporal_compression`` decoded frames; subsequent
        steps emit ``len_t * temporal_compression`` frames each.
        """
        if autoregressive_index == 0:
            return 1 + (self._len_t_latent - 1) * self._decoder_temporal_compression
        return self._len_t_latent * self._decoder_temporal_compression
