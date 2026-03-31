from abc import ABC, abstractmethod

from torch import Tensor


class BaseTextEncoder(ABC):
    @abstractmethod
    def encode(self, text: list[str]) -> Tensor:
        """
        Encode the batch of text into a tensor.

        Args:
            text: The batch of text to encode. [B]

        Returns:
            The encoded tensor. [B, L, D]
        """
        ...
