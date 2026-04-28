"""Pre-built :class:`Wan21PipelineConfig` builders for non-streaming Wan 2.1.

Currently shipped presets:

- ``"Wan2.1-T2V-1.3B"`` — official Wan-AI 1.3B T2V checkpoint paired
  with the Wan VAE decoder. CFG (negative prompt) is enabled via
  :attr:`WanTransformerConfig.guidance_scale`; the actual embedding
  strings are produced by the text encoder which lives inside
  :class:`Wan21Pipeline` (see :mod:`recipes.wan21.run`).

- ``"Wan2.1-I2V-14B-480P"`` — official Wan-AI 14B I2V checkpoint
  (480P preset). Wires:

  * the 14B :class:`WanDiTNetwork` with ``cross_attn_enable_img=True``
    (CLIP image features go through a per-block image cross-attention),
  * :attr:`WanTransformerConfig.concat_image_mask_to_latent` ``=True``
    (so the post-init bumps the network input channel count from 16 to
    36: ``[noisy_latent (16ch), mask (4ch), image_latent (16ch)]``),
  * the infra :class:`StreamInferencePipelineConfig.encoder` slot to a
    :class:`I2VCtrlEncoderConfig` (defined in
    :mod:`flashdreams.recipes.wan.autoencoder.i2v`) which produces the
    per-AR-step :class:`ImageCtrl`,
  * a long-lived :class:`CLIPImageEncoderConfig` on the project
    pipeline so the user-provided first frame is also encoded once
    into the cross-attention image features.

Batch / view / video resolution / per-chunk temporal length are
intentionally *not* exposed at the project layer: they are infra
concerns living on :class:`WanTransformerConfig` and are hardcoded to
canonical Wan 2.1 defaults inside this module. Callers that want to
deviate should construct :class:`Wan21PipelineConfig` directly.
"""

from __future__ import annotations


from flashdreams.infra.diffusion.model import DiffusionModelConfig
from flashdreams.infra.diffusion.scheduler import (
    FlowMatchUniPCSchedulerConfig,
)
from flashdreams.infra.encoder.image.clip import CLIPImageEncoderConfig
from flashdreams.recipes.wan.autoencoder.i2v import (
    I2VCtrlEncoderConfig,
)
from flashdreams.recipes.wan.autoencoder.vae import (
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
    WanVAEDecoderConfig,
    WanVAEEncoderConfig,
)
from flashdreams.recipes.wan.pipeline import WanInferencePipelineConfig
from flashdreams.recipes.wan.transformer.impl.network import (
    WanDiTNetwork14BConfig,
    WanDiTNetwork1pt3BConfig,
)
from flashdreams.recipes.wan.transformer.wan21 import Wan21TransformerConfig


def build_wan21_t2v_1pt3b_480p(
    *,
    video_height: int = 480,
    video_width: int = 832,
    len_t: int = 21,  # number of latent frames per AR chunk
    guidance_scale: float = 6.0,
    num_inference_steps: int = 50,
    shift: float = 8.0,
    seed: int = 42,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Wan 2.1 1.3B T2V pipeline config."""
    WAN_VAE_SPATIAL_COMPRESSION = 8

    latent_height = video_height // WAN_VAE_SPATIAL_COMPRESSION
    latent_width = video_width // WAN_VAE_SPATIAL_COMPRESSION

    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=None,
        decoder=WanVAEDecoderConfig(),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=Wan21TransformerConfig(
                network=WanDiTNetwork1pt3BConfig(),
                checkpoint_path=(
                    "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/"
                    "diffusion_pytorch_model.safetensors"
                ),
                batch_shape=(),
                height=latent_height,
                width=latent_width,
                len_t=len_t,
                window_size_t=len_t,
                guidance_scale=guidance_scale,
            ),
            scheduler=FlowMatchUniPCSchedulerConfig(
                num_inference_steps=num_inference_steps,
                shift=shift,
            ),
        ),
    )


def build_wan21_i2v_14b_480p(
    *,
    video_height: int = 480,
    video_width: int = 832,
    len_t: int = 21,  # number of latent frames per AR chunk
    guidance_scale: float = 5.0,
    num_inference_steps: int = 40,
    shift: float = 3.0,
    seed: int = 42,
    enable_sync_and_profile: bool = False,
) -> WanInferencePipelineConfig:
    """Wan 2.1 14B I2V."""
    WAN_VAE_SPATIAL_COMPRESSION = 8

    latent_height = video_height // WAN_VAE_SPATIAL_COMPRESSION
    latent_width = video_width // WAN_VAE_SPATIAL_COMPRESSION

    return WanInferencePipelineConfig(
        enable_sync_and_profile=enable_sync_and_profile,
        encoder=I2VCtrlEncoderConfig(
            encoder=WanVAEEncoderConfig(
                checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
            ),
        ),
        decoder=WanVAEDecoderConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        ),
        diffusion_model=DiffusionModelConfig(
            seed=seed,
            transformer=Wan21TransformerConfig(
                network=WanDiTNetwork14BConfig(
                    cross_attn_enable_img=True,
                ),
                checkpoint_path=(
                    "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P/blob/main/"
                    "diffusion_pytorch_model.safetensors.index.json"
                ),
                batch_shape=(),
                height=latent_height,
                width=latent_width,
                len_t=len_t,
                window_size_t=len_t,
                guidance_scale=guidance_scale,
                concat_image_mask_to_latent=True,
            ),
            scheduler=FlowMatchUniPCSchedulerConfig(
                num_inference_steps=num_inference_steps,
                shift=shift,
            ),
        ),
        image_encoder=CLIPImageEncoderConfig(
            model_id_or_local_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        ),
    )
