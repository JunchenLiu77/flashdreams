import torch
from torch import Tensor

from flashsim.model.text_encoder.base import BaseTextEncoder


class MockTextEncoder(BaseTextEncoder):
    """
    A mock text encoder for testing purposes.
    """

    def __init__(
        self,
        seq_len: int = 256,
        embedding_dim: int = 1024,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = torch.device("cuda"),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.device = device

    def encode(self, text: list[str]) -> Tensor:
        """
        Encode the batch of text into a tensor.

        Args:
            text: The batch of text to encode. [B]

        Returns:
            The encoded tensor. [B, seq_len, embedding_dim]
        """
        embeddings = torch.stack(
            [
                torch.randn(
                    self.seq_len,
                    self.embedding_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in text
            ]
        )
        return embeddings
