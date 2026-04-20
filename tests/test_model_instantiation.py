"""
Manual tests for model instantiation and checkpoint loading.

These tests require GPU and network access to download model weights.
Run with: pytest tests/test_model_instantiation.py -v -m manual

To run all tests including manual:
    pytest tests/test_model_instantiation.py -v
"""

import pytest
import torch

# Mark all tests in this module as manual (slow, require GPU/network)
pytestmark = [pytest.mark.manual, pytest.mark.slow]


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def dtype():
    return torch.bfloat16


class TestImageEncoder:
    """Tests for image encoders."""

    def test_wan_image_encoder_instantiation(self, device, dtype):
        """Test WanImageEncoder can be instantiated and encode images."""
        from flashsim.model.img_encoder.clip import WanImageEncoderConfig

        image_encoder = WanImageEncoderConfig().setup(device=device)

        image = torch.rand(1, 2, 3, 224, 224, device=device, dtype=dtype) * 2.0 - 1.0
        image_embeds = image_encoder.encode(image)

        assert image_embeds.shape == (1, 2, 257, 1280)
        assert image_embeds.dtype == dtype
        assert image_embeds.device.type == "cuda"


class TestTextEncoders:
    """Tests for text encoders."""

    def test_wan_text_encoder_instantiation(self, device):
        """Test WanTextEncoder can be instantiated and encode text."""
        from flashsim.model.text_encoder.wan2_1 import WanTextEncoderConfig

        text_encoder = WanTextEncoderConfig().setup()

        text = ["hello world"]
        text_embeddings = text_encoder.encode(text)

        assert text_embeddings.shape == (1, 512, 4096)
        assert text_embeddings.dtype == torch.bfloat16

    def test_cosmos_reason1_text_encoder_instantiation(self, device):
        """Test CosmosReason1TextEncoder can be instantiated and encode text."""
        from flashsim.model.text_encoder.cosmos_reason1 import (
            CosmosReason1TextEncoder,
            CosmosReason1TextEncoderConfig,
        )

        text_encoder = CosmosReason1TextEncoder(config=CosmosReason1TextEncoderConfig())

        text = ["A beautiful sunset over a calm ocean."]
        text_embeddings = text_encoder.encode(text)

        # full_concat strategy: 28 layers * 3584 hidden_size = 100352
        assert text_embeddings.shape == (1, 512, 100352)
        assert text_embeddings.dtype == torch.bfloat16
        assert text_embeddings.device.type == "cuda"


class TestVideoVAE:
    """Tests for video VAE models."""

    def test_pixel_shuffle_vae_instantiation(self, device):
        """Test PixelShuffleVAEInterface can be instantiated."""
        from flashsim.model.video_vae.pshuffle import PixelShuffleVAEInterfaceConfig

        model = PixelShuffleVAEInterfaceConfig().setup()

        assert model.temporal_compression_ratio == 4
        assert model.spatial_compression_ratio == 8

    def test_teahv_vae_instantiation(self, device):
        """Test TeahvInterface can be instantiated."""
        from flashsim.model.video_vae.teahv import TeahvInterfaceConfig

        model = TeahvInterfaceConfig().setup()

        assert model.temporal_compression_ratio == 4
        assert model.spatial_compression_ratio == 8

    def test_wan_vae_instantiation(self, device):
        """Test WanVAEInterface can be instantiated."""
        from flashsim.model.video_vae.wan import WanVAEInterfaceConfig

        model = WanVAEInterfaceConfig().setup()

        assert model.temporal_compression_ratio == 4
        assert model.spatial_compression_ratio == 8


class TestDiTNetwork:
    """Tests for DiT network models."""

    def test_wan_dit_t2v_1_3b_instantiation_and_checkpoint_loading(self, device):
        """Test WanDiTNetwork 1.3B T2V can be instantiated and load checkpoint."""
        from flashsim.model.video_dit.wan2_1.network import WanDiTNetwork1pt3BConfig
        from flashsim.checkpoint.load import load_checkpoint

        network_config = WanDiTNetwork1pt3BConfig()
        network = network_config.setup().to(device)

        state_dict = load_checkpoint(
            "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/diffusion_pytorch_model.safetensors"
        )
        network.load_state_dict(state_dict)

        assert network is not None

    def test_wan_dit_i2v_14b_instantiation_and_checkpoint_loading(self, device):
        """Test WanDiTNetwork 14B I2V can be instantiated and load checkpoint."""
        from flashsim.model.video_dit.wan2_1.network import WanDiTNetwork14BConfig
        from flashsim.checkpoint.load import load_checkpoint

        network_config = WanDiTNetwork14BConfig(
            cross_attn_enable_img=True, in_dim=16 + 20
        )
        network = network_config.setup().to(device)

        state_dict = load_checkpoint(
            "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/blob/main/diffusion_pytorch_model.safetensors.index.json"
        )
        network.load_state_dict(state_dict)

        assert network is not None
