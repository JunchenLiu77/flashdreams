from abc import ABC, abstractmethod

from typing import Any

import torch
from torch import Tensor


def denoise(noisy_input: Tensor, sigma: Tensor, predicted_flow: Tensor) -> Tensor:
    """
    Recover the clean input from the noisy input.

    Args:
        noisy_input: The noisy input tensor. [B, ...]
        sigma: The sigma. [1] or [B]
        predicted_flow: The predicted flow. Same shape as the input tensor `noisy_input`.

    Returns:
        The clean input tensor. Same shape as the input tensor `noisy_input`.
    """
    # broadcast sigma to the same shape as the noisy input
    sigma = sigma.view(-1, *([1] * (len(noisy_input.shape) - 1)))
    clean_input = noisy_input - sigma * predicted_flow
    return clean_input


def add_noise(
    clean_input: Tensor, sigma: Tensor, rng: torch.Generator | None = None
) -> Tensor:
    """
    Add noise to the clean input.

    Args:
        clean_input: The clean input tensor. [B, ...]
        sigma: The sigma. [1] or [B]
        rng: The random number generator to use for the noise.

    Returns:
        The noisy input tensor. Same shape as the input tensor `clean_input`.
    """
    # broadcast sigma to the same shape as the clean input
    sigma = sigma.view(-1, *([1] * (len(clean_input.shape) - 1)))
    noise = torch.randn_like(clean_input, generator=rng)
    noisy_input = (1.0 - sigma) * clean_input + sigma * noise
    return noisy_input


class BaseVideoDiT[VideoDiTCacheType](ABC):
    @abstractmethod
    def initialize_cache(self) -> VideoDiTCacheType:
        """
        Initialize the cache for DIT.
        """
        ...

    @abstractmethod
    def generate(
        self,
        condition: Any,
        cache: VideoDiTCacheType,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """
        Generation entrance for the video DiT.

        Args:
            condition: The condition for the video DiT.
            cache: The cache for the video DiT.
            rng: The random number generator to use for the noise.

        Returns:
            The generated tensor.
        """
        ...
