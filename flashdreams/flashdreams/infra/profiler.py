"""CUDA-event timer."""

from __future__ import annotations

import torch


class EventProfiler:
    """Times stages between a start event and one ``record(stage)`` per stage.

    Example::

        profiler = EventProfiler()              # records the start event
        run_encoder()
        profiler.record("encode")
        run_diffusion()
        profiler.record("diffuse")
        run_decoder()
        profiler.record("decode")

        stages_ms = profiler.sync_and_summarize()
        # -> {"encode": 12.3, "diffuse": 102.4, "decode": 45.6}
    """

    def __init__(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._ends: dict[str, torch.cuda.Event] = {}
        self._start.record()

    def record(self, stage: str) -> None:
        """Record an end-of-stage event under ``stage`` (must be unique)."""
        assert stage not in self._ends, f"stage {stage!r} already recorded"
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._ends[stage] = event

    def elapsed_ms(self) -> dict[str, float]:
        """Return ``{stage: ms}`` in record order (no sync)."""
        prev = self._start
        out: dict[str, float] = {}
        for stage, end in self._ends.items():
            out[stage] = prev.elapsed_time(end)
            prev = end
        return out

    def sync_and_summarize(self) -> dict[str, float]:
        """``torch.cuda.synchronize()`` then return :meth:`elapsed_ms`."""
        torch.cuda.synchronize()
        return self.elapsed_ms()
