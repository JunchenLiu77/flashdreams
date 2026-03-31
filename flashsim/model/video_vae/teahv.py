import math
import torch
from torch import Tensor

from flashsim.model.video_vae.impl.teahv import TAEHV, TAEHVCache
from flashsim.model.video_vae.base import BaseVideoVAE


class TeahvInterface(BaseVideoVAE[TAEHVCache, TAEHVCache]):
    def __init__(
        self,
        checkpoint_path: str,
        parallel: bool = True,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
        **kwargs,
    ):
        # parallel=True: faster + higher memory
        # parallel=False: lower memory
        self.parallel = parallel
        self.need_scaled = "lighttae" in checkpoint_path
        self.taehv = TAEHV(checkpoint_path, **kwargs).to(device=device, dtype=dtype)

        self.mean = torch.tensor(
            [
                -0.7571,
                -0.7089,
                -0.9113,
                0.1075,
                -0.1745,
                0.9653,
                -0.1517,
                1.5508,
                0.4134,
                -0.0715,
                0.5517,
                -0.3632,
                -0.1922,
                -0.9497,
                0.2503,
                -0.2921,
            ],
            dtype=dtype,
            device=device,
        )

        self.std = torch.tensor(
            [
                2.8184,
                1.4541,
                2.3275,
                2.6558,
                1.2196,
                1.7708,
                2.6052,
                2.0743,
                3.2687,
                2.1526,
                2.8652,
                1.5579,
                1.6382,
                1.1253,
                2.8251,
                1.9160,
            ],
            dtype=dtype,
            device=device,
        )

    def initialize_encode_cache(self) -> TAEHVCache:
        return self.taehv.prepare_cache()

    def encode(self, x: Tensor, cache: TAEHVCache | None = None) -> Tensor:
        raise NotImplementedError("Encoding is not supported for TeahvInterface")

    def initialize_decode_cache(self) -> TAEHVCache:
        return self.taehv.prepare_cache()

    def decode(self, z: Tensor, cache: TAEHVCache | None = None) -> Tensor:
        """
        z is expected to be in the format of [..., T, C, H, W]

        return: [..., T, C, H, W], values in range [-1, 1]
        """
        if cache is None:
            # create a temporary cache
            cache = self.initialize_encode_cache()

        assert z.ndim >= 4, "Expected input to have shape [..., T, C, H, W]"

        *batch_shape, T, C, H, W = z.shape
        batch_size = math.prod(batch_shape)
        z = z.reshape(batch_size, T, C, H, W)

        if self.need_scaled:
            z = z * self.std.view(1, 1, -1, 1, 1)
            z = z + self.mean.view(1, 1, -1, 1, 1)

        x = (
            self.taehv.decode_video(z, parallel=self.parallel, cache=cache)
            .mul_(2)
            .sub_(1)
        )
        return x.reshape(*batch_shape, *x.shape[1:])

    @property
    def temporal_compression_ratio(self) -> int:
        return 4

    @property
    def spatial_compression_ratio(self) -> int:
        return 8
