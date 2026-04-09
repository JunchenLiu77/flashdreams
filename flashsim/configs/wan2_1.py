from flashsim.pipeline.wan2_1 import Wan2_1PipelineConfig
from flashsim.model.video_vae.wan import (
    WanVAEInterfaceConfig,
    AVAILABLE_WAN_VAE_CHECKPOINT_PATHS,
)
from flashsim.model.text_encoder.wan2_1 import WanTextEncoderConfig
from flashsim.model.video_dit.wan2_1.model import (
    WanDiTConfig,
    AVAILABLE_WAN2_1_CHECKPOINT_PATHS,
)
from flashsim.model.video_dit.wan2_1.network import WanDiTNetworkConfig

WAN2_1_CONFIGS = {}

WAN2_1_CONFIGS["self_forcing"] = Wan2_1PipelineConfig(
    detokenizer=WanVAEInterfaceConfig(
        checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
    ),
    # detokenizer=TeahvInterfaceConfig(
    #     checkpoint_path=AVAILABLE_TAEHV_CHECKPOINT_PATHS["lighttae"],
    # ),
    text_encoder=WanTextEncoderConfig(),
    dit=WanDiTConfig(
        checkpoint_path=AVAILABLE_WAN2_1_CHECKPOINT_PATHS["self_forcing"],
        network=WanDiTNetworkConfig(
            patch_embedding_type="conv3d",
        ),
    ),
)
