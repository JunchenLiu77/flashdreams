"""Diffusion scheduler interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn as nn
from torch import Tensor

from flashdreams.infra.config import InstantiateConfig


class FlowPredictor(Protocol):
    """Closure ``(noisy_latent, timestep) -> predicted_flow``.

    Built by :class:`DiffusionModel.generate` by binding the per-AR-step
    ``cache`` / ``input`` and :meth:`Transformer.predict_flow`. A scheduler
    invokes it once per denoising iteration.

    Note: the scheduler decides the dtype of the ``timestep`` scalar --
    UniPC uses ``int64``, flow-match uses ``float`` -- and places it on
    the same device as ``noisy_latent``. The transformer must accept
    either dtype.
    """

    def __call__(self, noisy_latent: Tensor, timestep: Tensor) -> Tensor:
        """Predict the flow at ``timestep``.

        Args:
            noisy_latent: ``[...]`` tensor on the model's device/dtype
                (the scheduler passes its current iterate as-is; in
                practice ``[B, C, T, H, W]`` for video latents).
            timestep: 0-d tensor on the same device. dtype is
                scheduler-defined (see class note).

        Returns:
            Predicted flow with the same shape, device, and dtype as
            ``noisy_latent``.
        """
        ...


@dataclass(kw_only=True)
class SchedulerConfig(InstantiateConfig["Scheduler"]):
    """Hyperparameters for a :class:`Scheduler`."""

    _target: type["Scheduler"] = field(default_factory=lambda: Scheduler)

    num_inference_steps: int
    """Number of denoising iterations the scheduler runs in :meth:`sample`."""

    shift: float = 5.0
    """Schedule warp factor (family-specific; e.g. flow-match shift)."""


class Scheduler(nn.Module, ABC):
    """Denoising scheduler.

    Owns the entire denoising loop. Callers see only ``noise → clean``;
    the loop shape (renoise / multistep / plain ODE) is a private
    implementation detail.

    Example::

        scheduler = config.setup()
        clean = scheduler.sample(initial_noise=noise, predict_flow=predictor)
        # later, to corrupt a clean latent to a given timestep:
        noisy = scheduler.add_noise(clean_input=clean, timestep=t)
    """

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def sample(
        self,
        initial_noise: Tensor,
        predict_flow: FlowPredictor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Run the full denoising loop and return the clean latent.

        Schedulers are shape-agnostic: every internal op is elementwise
        broadcast against per-step scalar sigmas. In practice
        ``initial_noise`` is a video latent ``[B, C, T, H, W]``.

        Args:
            initial_noise: ``[...]`` Gaussian noise on the caller's
                device/dtype. The scheduler conventionally treats this
                as a sample at ``sigma=1`` (i.e. the highest noise
                level on its schedule).
            predict_flow: Per-step closure invoked exactly
                :attr:`SchedulerConfig.num_inference_steps` times.
            rng: ``torch.Generator`` on the same device as
                ``initial_noise``. Used by self-forcing renoise loops
                to draw extra noise per step; pure ODE solvers ignore
                it.

        Returns:
            ``[...]`` clean latent with the same shape, device, and
            dtype as ``initial_noise``.
        """

    @abstractmethod
    def add_noise(
        self,
        clean_input: Tensor,
        timestep: Tensor,
        rng: torch.Generator | None = None,
    ) -> Tensor:
        """Forward corruption ``x_t = (1 - sigma(t)) * x_0 + sigma(t) * eps``.

        Args:
            clean_input: ``[...]`` clean latent on the caller's
                device/dtype (typically ``[B, C, T, H, W]``).
            timestep: 0-d tensor on the same device. Scheduler-specific
                value semantics (FlowMatch snaps to the nearest entry
                of its 1000-step training table; FlowUniPC requires
                exact membership in its inference schedule).
            rng: ``torch.Generator`` on the same device as
                ``clean_input``, used to draw the additive Gaussian
                noise.

        Returns:
            ``[...]`` noisy latent with the same shape, device, and
            dtype as ``clean_input``.
        """
