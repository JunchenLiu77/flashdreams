"""Pre-built :class:`LingbotWorldInferencePipelineConfig` builders for streaming Lingbot World camera-control I2V.

Each builder maps a short name (``"LingBot-World-Fast"``,
``"LingBot-World-Fast-Flash"``) to a function taking only the
runtime knobs the recipe layer owns (``cp_size``, ``compile_network``,
``seed``, ``enable_sync_and_profile``) and returns a
fully-constructed :class:`LingbotWorldInferencePipelineConfig` — a
:class:`WanInferencePipelineConfig` subclass whose :meth:`generate`
accepts a :class:`CamCtrlInput` camera payload. The config differs
from the stock Wan 2.1 streaming configs in two slots:

- ``encoder`` is an :class:`I2VCamCtrlEncoderConfig`: a composite
  per-AR-step encoder that pairs a Wan-VAE-backed
  :class:`I2VCtrlEncoder` with a PixelShuffle pseudo-VAE over Plücker
  rays, consuming an :class:`I2VCamCtrlInput` (first-frame pixel
  chunk + per-AR-step intrinsics/poses/world-scale) and emitting an
  :class:`I2VCamCtrlEmbeddings` the transformer cross-attends to.
- ``transformer`` is a :class:`LingbotWorldTransformerConfig` over
  :class:`LingbotWorldDiTNetwork14BConfig` — a Wan 2.1 14B backbone
  with a per-block ``CamCtrlBlock`` that cross-attends to the
  Plücker volume.

Batch / view / video resolution / per-chunk temporal length are
intentionally *not* exposed at the recipe layer: they live on
:class:`LingbotWorldTransformerConfig` and are pinned to canonical
Lingbot World streaming defaults below. Callers that want to deviate
should construct :class:`LingbotWorldTransformerConfig` directly.
"""

from __future__ import annotations

from collections.abc import Callable

from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler.fm import FlowMatchSchedulerConfig
from flashdreams.recipes.alpadreams.encoder.pixel_shuffle import (
    PixelShuffleVAEEncoderConfig,
)
from flashdreams.recipes.lingbot_world.encoder.camctrl import (
    I2VCamCtrlEncoderConfig,
)
from flashdreams.recipes.lingbot_world.transformer import (
    LingbotWorldTransformerConfig,
)
from flashdreams.recipes.lingbot_world.transformer.impl.network import (
    LingbotWorldDiTNetwork14BConfig,
)
from flashdreams.recipes.taehv import TeahvVAEDecoderConfig
from flashdreams.recipes.wan.autoencoder.i2v import I2VCtrlEncoderConfig
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)
from flashdreams.recipes.lingbot_world.pipeline import (
    LingbotWorldInferencePipelineConfig,
)

AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS: dict[str, str] = {
    "LingBot-World-Fast": "https://huggingface.co/robbyant/lingbot-world-fast/blob/main/diffusion_pytorch_model.safetensors.index.json",
}


# ---------------------------------------------------------------------------
# Canonical Lingbot World streaming defaults
# ---------------------------------------------------------------------------

# Upstream Lingbot-World-Fast 4-step schedule and its 2-step Flash variant.
# Note the Flash list is not a prefix of the Fast one — they're
# independently distilled schedules.
_DEFAULT_DENOISING_TIMESTEPS = [999, 978, 947, 825]
_FLASH_DENOISING_TIMESTEPS = [999, 947]
_DEFAULT_NUM_TRAIN_TIMESTEPS = 1000

_DEFAULT_BATCH_SHAPE: tuple[int, ...] = (1, 1)  # [B=1, V=1]
_DEFAULT_VIDEO_HEIGHT = 464
_DEFAULT_VIDEO_WIDTH = 832
_DEFAULT_LEN_T_LATENT = 3
_WAN_VAE_SPATIAL_COMPRESSION = 8


def _wan_vae_decoder_config() -> WanVAEDecoderConfig:
    """Streaming Wan VAE decoder (4x temporal, 8x spatial upsample)."""
    return WanVAEDecoderConfig()


def _taehv_vae_decoder_config() -> TeahvVAEDecoderConfig:
    """Tiny AutoEncoder (TAEHV) decoder — drop-in faster replacement for Wan VAE."""
    return TeahvVAEDecoderConfig()


def _scheduler_config(
    denoising_timesteps: list[int],
) -> FlowMatchSchedulerConfig:
    """Lingbot World flow-match scheduler.

    Parameterized by the full timestep list rather than by
    ``num_inference_steps`` because the Fast (4-step) and Flash
    (2-step) variants ship *independently-distilled* schedules — the
    Flash list is not a prefix of the Fast list, so the
    ``[:num_inference_steps]`` shorthand used by
    :mod:`flashdreams.recipes.wan.config.causal_wan21` does not apply
    here.
    """
    return FlowMatchSchedulerConfig(
        num_inference_steps=len(denoising_timesteps),
        denoising_timesteps=denoising_timesteps,
        warp_denoising_step=False,
        shift=8.0,
        sigma_min=0.0,
        extra_one_step=True,
        num_train_timesteps=_DEFAULT_NUM_TRAIN_TIMESTEPS,
    )


def _transformer_config(
    *,
    checkpoint_path: str,
    cp_size: int,
    compile_network: bool,
) -> LingbotWorldTransformerConfig:
    """Lingbot World 14B transformer defaults for streaming inference.

    Shape knobs (batch / view / video resolution / latent T) are
    pinned to canonical Lingbot defaults; only the runtime knobs the
    caller owns (CP size, torch.compile toggle) are passed through.

    I2V conditioning is handled inside
    :meth:`LingbotWorldTransformer.predict_flow` by
    channel-concatenating the encoded first-frame latent and a
    4-channel binary mask onto the noisy latent
    (``concat_image_mask_to_latent=True``, ``stamp_image_latent=False``).
    Per-AR-step Plücker camera conditioning is routed through the
    per-block ``CamCtrlBlock`` via the ``plucker`` kwarg threaded in
    from the infra encoder by :class:`LingbotWorldTransformer`.
    """
    return LingbotWorldTransformerConfig(
        network=LingbotWorldDiTNetwork14BConfig(
            patch_embedding_type="conv3d",
            control_type="cam",
        ),
        checkpoint_path=checkpoint_path,
        batch_shape=_DEFAULT_BATCH_SHAPE,
        height=_DEFAULT_VIDEO_HEIGHT // _WAN_VAE_SPATIAL_COMPRESSION,
        width=_DEFAULT_VIDEO_WIDTH // _WAN_VAE_SPATIAL_COMPRESSION,
        len_t=_DEFAULT_LEN_T_LATENT,
        cp_size=cp_size,
        # CFG off by default to match the upstream Lingbot checkpoint.
        guidance_scale=1.0,
        # Streaming defaults.
        window_size_t=60,
        sink_size_t=0,
        # I2V channel-concat (mask + first-frame latent), not stamping.
        stamp_image_latent=False,
        concat_image_mask_to_latent=True,
        compile_network=compile_network,
    )


def _pipeline_encoder_config() -> I2VCamCtrlEncoderConfig:
    """Per-AR-step composite encoder slot.

    Always wired — Lingbot World is I2V-only and requires both a
    first frame and a camera stream. The composite runs:

    - a Wan-VAE-backed :class:`I2VCtrlEncoder` on the per-AR-step
      pixel chunk built from the user's first frame, and
    - a PixelShuffle pseudo-VAE on the Plücker ray volume rendered
      from the user's per-AR-step ``(intrinsics, poses, world_scale)``
      stream.

    Its :class:`I2VCamCtrlEmbeddings` output is forwarded to the
    transformer as the ``input`` argument.
    """
    return I2VCamCtrlEncoderConfig(
        i2v=I2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            ),
        ),
        plucker=PixelShuffleVAEEncoderConfig(
            frame_selection_mode="last_frame",
        ),
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_lingbot_world_fast(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    enable_sync_and_profile: bool = False,
) -> LingbotWorldInferencePipelineConfig:
    """LingBot-World-Fast checkpoint, Wan VAE decoder, 4-step distilled schedule."""
    return LingbotWorldInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(),
        decoder=_wan_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS[
                    "LingBot-World-Fast"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(_DEFAULT_DENOISING_TIMESTEPS),
        ),
    )


def build_lingbot_world_fast_flash(
    *,
    cp_size: int = 1,
    compile_network: bool = True,
    seed: int = 42,
    enable_sync_and_profile: bool = False,
) -> LingbotWorldInferencePipelineConfig:
    """LingBot-World-Fast checkpoint, TAEHV decoder, 2-step distilled schedule."""
    return LingbotWorldInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=_pipeline_encoder_config(),
        decoder=_taehv_vae_decoder_config(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=_transformer_config(
                checkpoint_path=AVAILABLE_LINGBOT_WORLD_CHECKPOINT_PATHS[
                    "LingBot-World-Fast"
                ],
                cp_size=cp_size,
                compile_network=compile_network,
            ),
            scheduler=_scheduler_config(_FLASH_DENOISING_TIMESTEPS),
        ),
    )


LINGBOT_WORLD_CONFIG_BUILDERS: dict[
    str, Callable[..., LingbotWorldInferencePipelineConfig]
] = {
    "LingBot-World-Fast": build_lingbot_world_fast,
    "LingBot-World-Fast-Flash": build_lingbot_world_fast_flash,
}
