from dataclasses import dataclass, field
import math
import torch
from torch import Tensor

from flashsim.model.video_vae.impl.wan import WanVAE, WanVAECache
from flashsim.model.video_vae.base import BaseVideoVAE
from flashsim.configs import InstantiateConfig


AVAILABLE_WAN_VAE_CHECKPOINT_PATHS = {
    "lightvae": "s3://flashsim/assets/checkpoints/autoencoders/lightvaew2_1.pth",
    "vae": "s3://flashsim/assets/checkpoints/autoencoders/Wan2.1_VAE.pth",
}


@dataclass
class WanVAEInterfaceConfig(InstantiateConfig["WanVAEInterface"]):
    _target: type["WanVAEInterface"] = field(default_factory=lambda: WanVAEInterface)

    checkpoint_path: str = AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"]

    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device("cuda")


class WanVAEInterface(BaseVideoVAE[WanVAECache, WanVAECache]):
    def __init__(self, config: WanVAEInterfaceConfig):
        use_lightvae = "lightvae" in config.checkpoint_path
        self.vae = WanVAE(
            vae_path=config.checkpoint_path,
            use_lightvae=use_lightvae,
            dtype=config.dtype,
            device=config.device,
        )

    def initialize_encode_cache(self) -> WanVAECache:
        return self.vae.prepare_cache()

    def encode(self, x: Tensor, cache: WanVAECache | None = None) -> Tensor:
        """
        x is expected to be in the format of [..., T, C, H, W], values in range [-1, 1]

        return: [..., T, C, H, W]
        """
        if cache is None:
            # create a temporary cache
            cache = self.initialize_encode_cache()

        assert x.ndim >= 4, "Expected input to have shape [..., T, C, H, W]"

        *batch_shape, T, C, H, W = x.shape
        batch_size = math.prod(batch_shape)
        x = x.reshape(batch_size, T, C, H, W)

        z = self.vae.encode(x.transpose(1, 2), cache=cache).transpose(1, 2)
        return z.reshape(*batch_shape, *z.shape[1:])

    def initialize_decode_cache(self) -> WanVAECache:
        return self.vae.prepare_cache()

    def decode(self, z: Tensor, cache: WanVAECache | None = None) -> Tensor:
        """
        z is expected to be in the format of [..., T, C, H, W]

        return: [..., T, C, H, W], values in range [-1, 1]
        """
        if cache is None:
            # create a temporary cache
            cache = self.initialize_decode_cache()

        assert z.ndim >= 4, "Expected input to have shape [..., T, C, H, W]"

        *batch_shape, T, C, H, W = z.shape
        batch_size = math.prod(batch_shape)
        z = z.reshape(batch_size, T, C, H, W)

        x = self.vae.decode(z.transpose(1, 2), cache=cache).transpose(1, 2)
        return x.reshape(*batch_shape, *x.shape[1:])

    @property
    def temporal_compression_ratio(self) -> int:
        return 4

    @property
    def spatial_compression_ratio(self) -> int:
        return 8


if __name__ == "__main__":
    import tyro

    config = tyro.cli(WanVAEInterfaceConfig)
    model = config.setup()
    print(model)
