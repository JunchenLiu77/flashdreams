from flashsim.configs import derive_conifg
from flashsim.pipeline.alpadreams import AlpadreamsPipelineConfig
from flashsim.model.video_vae.wan import (
    WanVAEInterfaceConfig,
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
)
from flashsim.model.video_vae.teahv import (
    TeahvInterfaceConfig,
    AVAILABLE_TAEHV_CHECKPOINT_PATHS,
)
from flashsim.model.text_encoder.cosmos_reason1 import CosmosReason1TextEncoderConfig
from flashsim.model.video_dit.alpadreams.model import (
    CosmosDiTConfig,
    AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS,
)
from flashsim.model.video_vae.pshuffle import PixelShuffleVAEInterfaceConfig

ALPADREAMS_CONFIGS = {}

ALPADREAMS_CONFIGS["sv_2steps_chunk2_loc6_lightvae_lighttae"] = (
    AlpadreamsPipelineConfig(
        tokenizer=WanVAEInterfaceConfig(
            checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
        ),
        detokenizer=TeahvInterfaceConfig(
            checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
        ),
        text_encoder=CosmosReason1TextEncoderConfig(),
        dit=CosmosDiTConfig(
            enable_hdmap_condition=True,
            encode_with_pixel_shuffle=False,
            num_views=1,
            # For 720P set to 3.0; for 480P set to 2.0;
            h_extrapolation_ratio=3.0,
            w_extrapolation_ratio=3.0,
            # Difussion schedule
            denoising_timesteps=[1000, 450],
            # Local attn: Number of tokens along T dimension.
            window_size_t=6,
            # Chunk size: Number of tokens along T dimension.
            len_t=2,
            # Checkpoint path
            checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"][
                "vae_encoding"
            ]["chunk2"],
        ),
    )
)

ALPADREAMS_CONFIGS["sv_2steps_chunk2_loc6_vae_vae"] = derive_conifg(
    ALPADREAMS_CONFIGS["sv_2steps_chunk2_loc6_lightvae_lighttae"],
    tokenizer=WanVAEInterfaceConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    detokenizer=WanVAEInterfaceConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
)


ALPADREAMS_CONFIGS["sv_2steps_chunk3_loc6_vae_vae"] = derive_conifg(
    ALPADREAMS_CONFIGS["sv_2steps_chunk2_loc6_lightvae_lighttae"],
    tokenizer=WanVAEInterfaceConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    detokenizer=WanVAEInterfaceConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    dit=dict(
        len_t=3,
        checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"][
            "vae_encoding"
        ]["chunk3"],
    ),
)

ALPADREAMS_CONFIGS["sv_2steps_chunk4_loc8_pshuffle_lighttae"] = derive_conifg(
    ALPADREAMS_CONFIGS["sv_2steps_chunk2_loc6_lightvae_lighttae"],
    tokenizer=PixelShuffleVAEInterfaceConfig(),
    detokenizer=TeahvInterfaceConfig(
        checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
    ),
    dit=CosmosDiTConfig(
        enable_hdmap_condition=True,
        encode_with_pixel_shuffle=True,
        # For 720P set to 3.0; for 480P set to 2.0;
        h_extrapolation_ratio=3.0,
        w_extrapolation_ratio=3.0,
        # Difussion schedule
        denoising_timesteps=[1000, 450],
        # Local attn: Number of tokens along T dimension.
        window_size_t=8,
        # Chunk size: Number of tokens along T dimension.
        len_t=4,
        # Checkpoint path
        checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"][
            "pixel_shuffle"
        ],
    ),
)


ALPADREAMS_CONFIGS["mv_2steps_chunk4_loc8_pshuffle_lighttae"] = (
    AlpadreamsPipelineConfig(
        tokenizer=PixelShuffleVAEInterfaceConfig(),
        detokenizer=TeahvInterfaceConfig(
            checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
        ),
        text_encoder=CosmosReason1TextEncoderConfig(),
        dit=CosmosDiTConfig(
            enable_hdmap_condition=True,
            encode_with_pixel_shuffle=True,
            num_views=4,
            # For 720P set to 3.0; for 480P set to 2.0;
            h_extrapolation_ratio=3.0,
            w_extrapolation_ratio=3.0,
            # Difussion schedule
            denoising_timesteps=[1000, 450],
            # Local attn: Number of tokens along T dimension.
            window_size_t=8,
            # Chunk size: Number of tokens along T dimension.
            len_t=4,
            # Checkpoint path
            checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["4views"][
                "pixel_shuffle"
            ],
        ),
    )
)
