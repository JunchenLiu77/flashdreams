from flashsim.pipeline.wan2_2 import Wan2_2PipelineConfig
from flashsim.model.video_vae.wan import (
    WanVAEInterfaceConfig,
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
)
from flashsim.model.text_encoder.wan2_1 import WanTextEncoderConfig
from flashsim.model.video_dit.wan2_2.model import (
    WanDiTConfig,
    AVAILABLE_WAN2_2_CHECKPOINT_PATHS,
)
from flashsim.model.video_dit.wan2_1.network import WanDiTNetwork14BConfig

WAN2_2_CONFIGS = {}

WAN2_2_CONFIGS["fastvideo-i2v"] = Wan2_2PipelineConfig(
    detokenizer=WanVAEInterfaceConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    text_encoder=WanTextEncoderConfig(),
    dit=WanDiTConfig(
        checkpoint_path_high_noise=AVAILABLE_WAN2_2_CHECKPOINT_PATHS["fastvideo-i2v"][
            "high_noise"
        ],
        checkpoint_path_low_noise=AVAILABLE_WAN2_2_CHECKPOINT_PATHS["fastvideo-i2v"][
            "low_noise"
        ],
        network_high_noise=WanDiTNetwork14BConfig(
            patch_embedding_type="conv3d",
        ),
        network_low_noise=WanDiTNetwork14BConfig(
            patch_embedding_type="conv3d",
        ),
    ),
)
