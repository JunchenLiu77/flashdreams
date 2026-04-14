from dataclasses import dataclass, field
from typing import Literal
import math
import os
import torch
from torch import Tensor

from transformers import CLIPImageProcessor, CLIPVisionModel

from flashsim.configs import InstantiateConfig


@dataclass
class WanImageEncoderConfig(InstantiateConfig["WanImageEncoder"]):
    _target: type["WanImageEncoder"] = field(default_factory=lambda: WanImageEncoder)

    model_id_or_local_path: Literal[
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ] = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    dtype: torch.dtype = torch.bfloat16


class WanImageEncoder:
    def __init__(
        self, config: WanImageEncoderConfig, device: torch.device = torch.device("cuda")
    ):
        # image encoder
        self.image_encoder = CLIPVisionModel.from_pretrained(
            config.model_id_or_local_path,
            cache_dir=os.getenv("HF_HOME", None),
            subfolder="image_encoder",
            dtype=config.dtype,
        )
        self.image_encoder.to(device)
        self.image_encoder.eval().requires_grad_(False)

        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.model_id_or_local_path,
            cache_dir=os.getenv("HF_HOME", None),
            subfolder="image_processor",
        )

    def encode(self, images: Tensor) -> Tensor:
        """
        Encode the images using the image encoder.

        Args:
            images: The images to encode. [..., C, H, W], in the range [-1, 1]

        Returns:
            The image embeddings. [..., 257, 1280]
        """
        batch_shape = images.shape[:-3]
        batch_size = math.prod(batch_shape)
        images = images.reshape(batch_size, *images.shape[-3:])

        device = self.image_encoder.device
        images = (images + 1) / 2.0
        images = self.image_processor(
            images=images.to(dtype=torch.float32), return_tensors="pt", do_rescale=False
        ).to(device, dtype=self.image_encoder.dtype)
        image_embeds = self.image_encoder(**images, output_hidden_states=True)

        output = image_embeds.hidden_states[-2]
        output = output.reshape(*batch_shape, *output.shape[-2:])
        return output


# python -m flashsim.model.img_encoder.clip
if __name__ == "__main__":
    device = torch.device("cuda")
    dtype = torch.bfloat16

    image_encoder = WanImageEncoderConfig().setup(device=device)

    image = torch.rand(1, 2, 3, 224, 224, device=device, dtype=dtype) * 2.0 - 1.0
    image_embeds = image_encoder.encode(image)

    print(image_embeds.shape)  # torch.Size([1, 2, 257, 1280])
    print(image_embeds.dtype)
    print(image_embeds.device)
    print(image_embeds.sum())  # tensor(23040., device='cuda:0', dtype=torch.bfloat16)
