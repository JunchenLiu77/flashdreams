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

"""Pre-built :class:`WanInferencePipelineConfig` builders for streaming Wan 2.2.

Each builder maps a short name (``"fastvideo"``, ...) to a function that
takes only the runtime knobs the recipe layer must own (``cp_size``,
``compile_network``, ``seed``) and returns a fully-constructed
:class:`WanInferencePipelineConfig` that uses the Wan 2.2 MoE transformer
(two Wan 2.1 14B networks with timestep-based dispatch) instead of the
Wan 2.1 backbone used by
:mod:`flashdreams.recipes.wan.config.causal_wan21`. Only T2V is supported
today (I2V with the FastVideo checkpoint uses a first-frame VAE-seed
warmup that doesn't fit the unified pipeline's per-AR-step
mask-injection I2V).

Dual high-noise / low-noise 14B networks load from the upstream HF repo
and are remapped to the official Wan key layout via
:data:`flashdreams.recipes.wan.transformer.wan22.CHECKPOINT_KEY_MAPPING`.

Batch / video resolution / per-chunk temporal length are intentionally
*not* exposed at the recipe layer: they live on
:class:`Wan21TransformerConfig` (the per-branch sub-config) and are
hardcoded to canonical Wan 2.2 streaming defaults inside this module.
Callers that want to deviate should construct
:class:`Wan22TransformerConfig` directly.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from flashdreams.core.checkpoint.remap import remap_checkpoint_keys
from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork14BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig
from flashdreams.recipes.wan.transformer.wan22 import (
    CHECKPOINT_KEY_MAPPING,
    Wan22TransformerConfig,
)

AVAILABLE_CAUSAL_WAN22_CHECKPOINT_PATHS: dict[str, dict[str, str]] = {
    "fastvideo": {
        "high_noise": "https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers/blob/main/transformer/diffusion_pytorch_model.safetensors",
        "low_noise": "https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers/blob/main/transformer_2/diffusion_pytorch_model.safetensors",
    },
}


# ---------------------------------------------------------------------------
# Checkpoint remap
# ---------------------------------------------------------------------------


def _remap_diffusers_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap the HF diffusers Wan 2.2 transformer state-dict to the bare
    ``WanDiTNetwork`` layout expected by :class:`Wan21Transformer`."""
    return remap_checkpoint_keys(state_dict, CHECKPOINT_KEY_MAPPING)


# ---------------------------------------------------------------------------
# Canonical Wan 2.2 streaming defaults
# ---------------------------------------------------------------------------

# FastVideo 8-step distillation schedule.
_DEFAULT_DENOISING_TIMESTEPS = [1000, 850, 700, 550, 350, 275, 200, 125]
_DEFAULT_NUM_TRAIN_TIMESTEPS = 1000
_DEFAULT_BOUNDARY_RATIO = 0.875

_DEFAULT_BATCH_SHAPE: tuple[int, ...] = (1,)
_DEFAULT_VIDEO_HEIGHT = 480
_DEFAULT_VIDEO_WIDTH = 832
_DEFAULT_LEN_T_LATENT = 3
_WAN_VAE_SPATIAL_COMPRESSION = 8


def _wan_vae_decoder_config() -> WanVAEDecoderConfig:
    """Wan VAE decoder config."""
    return WanVAEDecoderConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    )


def _scheduler_config(
    num_inference_steps: int = len(_DEFAULT_DENOISING_TIMESTEPS),
) -> FlowMatchSchedulerConfig:
    """FastVideo Wan 2.2 flow-match scheduler defaults."""
    timesteps = _DEFAULT_DENOISING_TIMESTEPS[:num_inference_steps]
    return FlowMatchSchedulerConfig(
        num_inference_steps=num_inference_steps,
        denoising_timesteps=timesteps,
        warp_denoising_step=True,
        shift=5.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=_DEFAULT_NUM_TRAIN_TIMESTEPS,
    )


def _transformer_config(
    *,
    checkpoint_path: dict[str, str],
    cp_size: int,
    compile_network: bool,
) -> Wan22TransformerConfig:
    """Wan 2.2 dual-14B transformer defaults for causal/streaming T2V inference.

    Shape knobs (batch / video height / video width) are hardcoded to the
    canonical Wan 2.2 streaming defaults; only the runtime knobs the
    caller actually owns (CP size, torch.compile toggle) are exposed.
    Both branches share those shape defaults (required by
    :class:`Wan22TransformerConfig.__post_init__`).
    """

    def _branch(ckpt: str) -> Wan21TransformerConfig:
        return Wan21TransformerConfig(
            network=WanDiTNetwork14BConfig(
                patch_embedding_type="conv3d",
            ),
            checkpoint_path=ckpt,
            state_dict_transform=_remap_diffusers_state_dict,
            batch_shape=_DEFAULT_BATCH_SHAPE,
            height=_DEFAULT_VIDEO_HEIGHT // _WAN_VAE_SPATIAL_COMPRESSION,
            width=_DEFAULT_VIDEO_WIDTH // _WAN_VAE_SPATIAL_COMPRESSION,
            len_t=_DEFAULT_LEN_T_LATENT,
            cp_size=cp_size,
            # CFG off: FastVideo's distilled checkpoint is a single
            # conditional forward.
            guidance_scale=1.0,
            # Streaming defaults.
            window_size_t=21,
            sink_size_t=0,
            compile_network=compile_network,
        )

    return Wan22TransformerConfig(
        transformer_high_noise=_branch(checkpoint_path["high_noise"]),
        transformer_low_noise=_branch(checkpoint_path["low_noise"]),
        boundary_ratio=_DEFAULT_BOUNDARY_RATIO,
        num_train_timesteps=_DEFAULT_NUM_TRAIN_TIMESTEPS,
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_fastvideo(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """FastVideo CausalWan2.2 distilled checkpoint with the Wan VAE decoder.

    T2V only. Structural twin of
    :func:`flashdreams.recipes.wan.config.causal_wan21.build_self_forcing`
    with the Wan 2.2 MoE transformer swapped in for the Wan 2.1 backbone.
    """
    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_CAUSAL_WAN22_CHECKPOINT_PATHS["fastvideo"],
                cp_size=cp_size,
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(),
        ),
    )


CAUSAL_WAN22_CONFIG_BUILDERS: dict[str, Callable[..., WanInferencePipelineConfig]] = {
    "fastvideo": build_fastvideo,
}
