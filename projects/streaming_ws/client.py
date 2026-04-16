"""WebSocket client: FRME in, CTRL out, clock-driven playout + latency profile.

Pipeline (after connect)::

    recv_loop  -> incoming Queue (batch_id, frames, recv_mono)
    sender_loop -> periodic CTRL (overlaps server work with local playout)
    run_playout -> fixed-FPS ticks; optional on_frame (console log and/or OpenCV window)

``recv_mono`` pairs with server ``batch_id`` so the client can measure
queue+playout delay for the first frame of each batch.

**Visual preview (laptop):** install ``opencv-python`` (extra ``streaming_viewer``) and pass
``--show-window``. Requires a local display (X11/Wayland on Linux, desktop on macOS/Windows);
headless SSH without forwarding will not show a window.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from projects.streaming_ws.playout import PlayoutConfig, run_playout
from projects.streaming_ws.protocol import pack_ctrl, unpack_frme


@dataclass(frozen=True)
class ClientConfig:
    uri: str = "ws://127.0.0.1:8765"
    frames_per_batch: int = 8
    target_fps: float = 60.0
    min_batches_before_start: int = 2
    playout_queue_batches: int = 4
    # None = run until Ctrl+C (SIGINT); otherwise bounded seconds.
    duration_s: float | None = None
    drop_stale_batches: bool = False
    send_ahead_factor: float = 0.92
    max_ws_message_bytes: int = 64 * 1024 * 1024
    verbose_profile: bool = True
    # Log one line per mock display tick (UTC wall clock + batch / frame).
    log_every_frame: bool = True
    # Pop an OpenCV window and decode each WebP (needs ``pip install .[streaming_viewer]``).
    show_window: bool = False
    window_name: str = "streaming_ws"


def _ensure_opencv() -> None:
    try:
        import cv2  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "show_window=True requires opencv-python. "
            "Install: pip install 'opencv-python>=4.8' "
            "or pip install -e '.[streaming_viewer]'"
        ) from e


def _make_on_display(*, frames_per_batch: int) -> Any:
    """Build a playout callback that logs each mock UI tick (real frame vs held repeat)."""
    seq = 0

    async def on_display(
        *,
        batch_id: int,
        frame_index: int,
        jpeg: bytes,
        held: bool,
    ) -> None:
        nonlocal seq
        seq += 1
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        wall = time.time()
        if held:
            line = (
                f"[DISPLAY] seq={seq} ts_utc={ts} wall_unix={wall:.6f} "
                f"kind=held_repeat batch=-1 frame=-1 bytes={len(jpeg)}"
            )
        else:
            line = (
                f"[DISPLAY] seq={seq} ts_utc={ts} wall_unix={wall:.6f} "
                f"kind=received_frame batch={batch_id} "
                f"frame={frame_index + 1}/{frames_per_batch} bytes={len(jpeg)}"
            )
        print(line, flush=True)

    return on_display


def _make_on_frame(
    cfg: ClientConfig,
    *,
    window_created: list[bool],
) -> Any | None:
    """Combine optional console logging and optional OpenCV preview (same playout tick)."""
    log_fn = (
        _make_on_display(frames_per_batch=cfg.frames_per_batch)
        if cfg.log_every_frame
        else None
    )
    if log_fn is None and not cfg.show_window:
        return None

    async def on_frame(
        *,
        batch_id: int,
        frame_index: int,
        jpeg: bytes,
        held: bool,
    ) -> None:
        if log_fn is not None:
            await log_fn(
                batch_id=batch_id, frame_index=frame_index, jpeg=jpeg, held=held
            )
        if cfg.show_window:
            from projects.streaming_ws.opencv_viewer import show_webp_in_window

            await asyncio.to_thread(
                show_webp_in_window,
                jpeg,
                window_name=cfg.window_name,
                window_created=window_created,
            )

    return on_frame


async def run_client(cfg: ClientConfig) -> dict[str, Any]:
    """Run one session; return stats dict including ``profile`` when verbose."""
    # Third tuple element: perf_counter() right after unpack (batch receive time).
    incoming: asyncio.Queue[tuple[int, tuple[bytes, ...], float]] = asyncio.Queue(
        maxsize=max(1, cfg.playout_queue_batches)
    )
    # Set once we have enough FRME batches to start playout (see min_batches_before_start).
    primed = asyncio.Event()
    # Stops sender_loop after playout returns (separate from SIGINT playout stop).
    stop = asyncio.Event()

    # Slightly faster than strict batch/ FPS so the server often has work queued ahead.
    send_interval = max(
        (cfg.frames_per_batch / max(cfg.target_fps, 1e-6)) * cfg.send_ahead_factor,
        1e-3,
    )

    async def recv_loop(ws: Any) -> None:
        try:
            async for message in ws:
                if not isinstance(message, (bytes, bytearray)):
                    continue
                recv_mono = time.perf_counter()
                frme = unpack_frme(bytes(message))
                item = (frme.batch_id, frme.frames, recv_mono)
                try:
                    incoming.put_nowait(item)
                except asyncio.QueueFull:
                    # Drop oldest batch to cap memory; pair with playout drop_stale_batches if needed.
                    with contextlib.suppress(asyncio.QueueEmpty):
                        incoming.get_nowait()
                    incoming.put_nowait(item)
                if incoming.qsize() >= cfg.min_batches_before_start:
                    primed.set()
        except ConnectionClosed:
            pass

    async def sender_loop(ws: Any) -> None:
        seq = 0
        await ws.send(pack_ctrl(seq, {"phase": "start", "t": time.monotonic()}))
        seq = 1
        while not stop.is_set():
            await asyncio.sleep(send_interval)
            try:
                await ws.send(pack_ctrl(seq, {"seq": seq, "t": time.monotonic()}))
            except ConnectionClosed:
                break
            seq += 1

    # SIGINT ends playout cleanly (avoids hanging forever when duration_s is None).
    playout_stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    sig_installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, playout_stop.set)
        sig_installed = True
    except (ValueError, NotImplementedError, RuntimeError):
        # e.g. non-main thread: rely on KeyboardInterrupt + duration_s instead.
        pass

    if cfg.show_window:
        _ensure_opencv()
    window_created = [False]
    on_frame = _make_on_frame(cfg, window_created=window_created)

    t0 = time.perf_counter()
    try:
        async with websockets.connect(
            cfg.uri,
            compression=None,
            max_size=cfg.max_ws_message_bytes,
        ) as ws:
            recv_task = asyncio.create_task(recv_loop(ws))
            try:
                await asyncio.wait_for(primed.wait(), timeout=120.0)
            except asyncio.TimeoutError:
                recv_task.cancel()
                raise RuntimeError(
                    f"did not receive {cfg.min_batches_before_start} batches within timeout"
                ) from None

            sender_task = asyncio.create_task(sender_loop(ws))
            play_cfg = PlayoutConfig(
                target_fps=cfg.target_fps,
                min_batches_before_start=cfg.min_batches_before_start,
                duration_s=cfg.duration_s,
                drop_stale_batches=cfg.drop_stale_batches,
                stop_event=playout_stop,
            )
            stats = await run_playout(incoming, cfg=play_cfg, on_frame=on_frame)
            stop.set()  # let sender_loop exit its sleep loop
            sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender_task
            await ws.close()
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recv_task
    finally:
        if cfg.show_window:
            from projects.streaming_ws.opencv_viewer import destroy_viewer_window

            destroy_viewer_window()
        if sig_installed:
            with contextlib.suppress(ValueError, NotImplementedError, RuntimeError):
                loop.remove_signal_handler(signal.SIGINT)

    elapsed = time.perf_counter() - t0
    play_elapsed = max(elapsed, 1e-9)
    out: dict[str, Any] = {
        "elapsed_s": elapsed,
        "frames_emitted": stats.frames_emitted,
        "underruns": stats.underruns,
        "batches_started": stats.batches_started,
        "fps_effective": stats.frames_emitted / play_elapsed,
    }
    if cfg.verbose_profile:
        prof = stats.profile_summary(target_fps=cfg.target_fps)
        out["profile"] = prof
        tgt = float(prof["target_frame_interval_ms"])
        p95_iv = float(prof["interval_p95_ms"])
        mean_iv = float(prof["interval_mean_ms"])
        # Clock can stay regular even while repeating one texture (held underruns).
        out["smooth_display_clock"] = (
            not math.isnan(mean_iv)
            and abs(mean_iv - tgt) <= 0.15 * tgt
            and not math.isnan(p95_iv)
            and p95_iv <= 1.35 * tgt
        )
        # True only if every tick showed a newly decoded frame (no starve/hold path).
        out["smooth_no_buffer_starve"] = stats.underruns == 0
    return out


def main_client(cfg: ClientConfig) -> None:
    if cfg.duration_s is None:
        print(
            "[client] duration_s=None — run until Ctrl+C (SIGINT); "
            "summary prints after clean stop.",
            flush=True,
        )
    if cfg.show_window:
        print(
            f"[client] OpenCV window '{cfg.window_name}' — close window or Ctrl+C to stop.",
            flush=True,
        )
    try:
        stats = asyncio.run(run_client(cfg))
    except KeyboardInterrupt:
        print("\n[client] Ctrl+C (KeyboardInterrupt) — exiting.", flush=True)
        sys.exit(130)

    print(
        f"elapsed={stats['elapsed_s']:.2f}s "
        f"frames={stats['frames_emitted']} "
        f"underruns={stats['underruns']} "
        f"batches_started={stats['batches_started']} "
        f"fps_effective~={stats['fps_effective']:.1f}"
    )
    if cfg.verbose_profile and "profile" in stats:
        p = stats["profile"]
        print(
            f"playout_target={p['target_fps']:.0f}fps "
            f"target_dt={p['target_frame_interval_ms']:.2f}ms "
            f"interval_mean={p['interval_mean_ms']:.2f}ms "
            f"p50={p['interval_p50_ms']:.2f}ms "
            f"p95={p['interval_p95_ms']:.2f}ms "
            f"max={p['interval_max_ms']:.2f}ms"
        )
        print(
            f"batch_recv_to_first_frame_ms "
            f"mean={p['batch_to_first_frame_mean_ms']:.1f} "
            f"p95={p['batch_to_first_frame_p95_ms']:.1f} "
            f"max={p['batch_to_first_frame_max_ms']:.1f}"
        )
        clk = "yes" if stats.get("smooth_display_clock") else "no"
        buf = "yes" if stats.get("smooth_no_buffer_starve") else "no"
        print(
            f"display_clock_smooth_{int(p['target_fps'])}fps={clk} "
            f"(mean/p95 frame intervals near {p['target_frame_interval_ms']:.2f}ms); "
            f"no_buffer_starve={buf} (underruns={stats['underruns']})"
        )
