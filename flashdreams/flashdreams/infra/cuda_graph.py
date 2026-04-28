"""Reusable CUDA-graph capture wrapper for stateful inference callables."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch


class CUDAGraphWrapper:
    """Capture a stateful callable's CUDA execution into a graph.

    The callable runs eagerly for ``warmup_iters`` calls (kernels JIT-load
    and the caching allocator stabilises), then the next call captures
    the whole forward into a :class:`torch.cuda.CUDAGraph` against a
    static input buffer; every subsequent same-shape call copies the
    new input into that buffer and replays the graph, returning a
    clone of the static output (the next replay would overwrite it).

    The captured kernels reference the static input pointer, the
    callable's parameters, the output buffer pointer, AND any internal
    mutable buffers the callable touches in place (e.g. cache slots).
    Those internal buffers must be at stable storage addresses across
    calls -- typically by writing through an in-place ``copy_`` once a
    slot's shape stabilises during warmup.

    A shape change drops the captured graph and restarts the cycle;
    :meth:`reset` does the same explicitly (call when the external
    state the callable depends on -- e.g. a fresh streaming cache --
    is swapped out).

    Compatibility with ``torch.compile``: ``fn`` may be a compiled
    callable, but Inductor + triton trigger lazy autotunes
    (``torch.cuda.synchronize()`` calls, illegal during graph capture)
    on the first call seen for each input shape. Callers MUST drain
    those autotunes through :meth:`drain` (or an unwrapped invocation
    with the same shape on the bare compiled callable) before the
    wrapped path attempts capture -- otherwise capture fails with
    ``cudaErrorStreamCaptureUnsupported``.

    Example::

        wrapper = CUDAGraphWrapper(model.forward, warmup_iters=2)
        # Rollout 1 -- variable shapes / Inductor autotune drain.
        for chunk in first_rollout_chunks:
            y = wrapper.drain(chunk)        # eager, shared static buffer
        # Rollout 2+ -- steady shape, capture + replay.
        for chunk in steady_rollout_chunks:
            y = wrapper(chunk)              # 2 warmups -> capture -> replays
    """

    def __init__(self, fn: Callable[..., torch.Tensor], warmup_iters: int = 2):
        self.fn = fn
        self.warmup_iters = warmup_iters
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_input: Optional[torch.Tensor] = None
        self._static_output: Optional[torch.Tensor] = None
        self._warmup_remaining = warmup_iters

    def reset(self) -> None:
        self._graph = None
        self._static_input = None
        self._static_output = None
        self._warmup_remaining = self.warmup_iters

    def _stage_input(self, x: torch.Tensor) -> torch.Tensor:
        """Copy ``x`` into the (lazily-allocated, contiguous) static input
        buffer and return that buffer. Used by every entry point so
        ``self.fn`` always sees the same stride pattern."""
        if self._static_input is None or self._static_input.shape != x.shape:
            self.reset()
            self._static_input = torch.empty_like(x).contiguous()
        self._static_input.copy_(x)
        return self._static_input

    def drain(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Eager autotune drain through the shared static buffer.

        Use during the very first rollout (variable-shape edges aside)
        so Inductor's lazy triton autotunes -- which call
        ``torch.cuda.synchronize`` and would crash a graph capture --
        run on the eager path against the SAME buffer + stride that
        :meth:`__call__` will later capture against. This prevents a
        second Inductor specialisation (and a second multi-second
        autotune) when the wrapper takes over from rollout 2 onwards.

        Doesn't consume ``warmup_iters`` and doesn't capture.
        """
        return self.fn(self._stage_input(x), *args, **kwargs)

    def __call__(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        static_input = self._stage_input(x)

        if self._graph is not None:
            self._graph.replay()
            # Clone: replay-on-next-call would overwrite the buffer.
            return self._static_output.clone()

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            return self.fn(static_input, *args, **kwargs)

        # Capture: trace one full forward against the static buffers.
        # `cudaStreamBeginCapture` only RECORDS the kernels -- it does
        # not execute them -- so the static output / any in-place cache
        # updates are no-ops at this point. Replay once immediately to
        # actually compute the output and advance the cache.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self.fn(static_input, *args, **kwargs)
        self._graph.replay()
        return self._static_output.clone()
