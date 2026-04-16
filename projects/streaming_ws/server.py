"""Asyncio WebSocket server: CTRL in, batched FRME out.

Defaults match a **mock cloud renderer**: 1280×720 WebP batches of 8 frames, plus a
fixed **200 ms** ``asyncio.sleep`` per CTRL-driven batch (prefill batches skip the
sleep). Tune ``--stub-latency-ms``, ``--frame-width``, ``--frame-height`` for other
scenarios.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from dataclasses import dataclass

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from projects.streaming_ws.protocol import pack_frme, unpack_ctrl
from projects.streaming_ws.stub_frames import encode_stub_batch

# Producer waits on ``await control_q.get()``; when the socket closes, the reader
# must unblock it. ``_SENTINEL`` is a unique object we push after draining stale CTRLs.
_SENTINEL = object()


@dataclass(frozen=True)
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    frames_per_batch: int = 8
    # Default mock: 720p batches with a fixed "inference" delay before encode+send.
    frame_width: int = 1280
    frame_height: int = 720
    prefill_batches: int = 2
    stub_latency_ms: float = 800.0
    max_ws_message_bytes: int = 64 * 1024 * 1024


async def _reader(control_q: asyncio.Queue, conn: ServerConnection) -> None:
    """Decode CTRL messages; keep only the **latest** control (interactive coalescing)."""
    try:
        async for message in conn:
            if not isinstance(message, (bytes, bytearray)):
                continue
            try:
                cm = unpack_ctrl(bytes(message))
            except ValueError:
                continue
            try:
                control_q.put_nowait(cm)
            except asyncio.QueueFull:
                # Queue size 1: drop the previous CTRL so gameplay stays on newest input.
                with contextlib.suppress(asyncio.QueueEmpty):
                    control_q.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    control_q.put_nowait(cm)
    finally:
        # Wake the producer if it is blocked on control_q.get() after disconnect.
        while True:
            try:
                control_q.get_nowait()
            except asyncio.QueueEmpty:
                break
        await control_q.put(_SENTINEL)


async def _handle_connection(conn: ServerConnection, cfg: ServerConfig) -> None:
    """Send prefill FRMEs (no CTRL needed), then one FRME per latest CTRL until close."""
    control_q: asyncio.Queue = asyncio.Queue(maxsize=1)
    # Reader runs concurrently so CTRL can arrive while we encode/send prefill.
    reader = asyncio.create_task(_reader(control_q, conn))
    batch_id = 0
    base_frame = 0
    try:
        # Warm the client buffer before first real CTRL/RTT (no stub_latency sleep here).
        for _ in range(cfg.prefill_batches):
            frames = encode_stub_batch(
                ctrl=None,
                batch_id=batch_id,
                width=cfg.frame_width,
                height=cfg.frame_height,
                n_frames=cfg.frames_per_batch,
                base_frame=base_frame,
            )
            base_frame += cfg.frames_per_batch
            blob = pack_frme(
                n_frames=cfg.frames_per_batch,
                width=cfg.frame_width,
                height=cfg.frame_height,
                batch_id=batch_id,
                frames=frames,
            )
            await conn.send(blob)
            batch_id += 1

        while True:
            item = await control_q.get()
            if item is _SENTINEL:
                break
            ctrl = item
            # Mock GPU/inference time before building the next 8-frame WebP batch.
            if cfg.stub_latency_ms > 0:
                await asyncio.sleep(cfg.stub_latency_ms / 1000.0)
            frames = encode_stub_batch(
                ctrl=ctrl,
                batch_id=batch_id,
                width=cfg.frame_width,
                height=cfg.frame_height,
                n_frames=cfg.frames_per_batch,
                base_frame=base_frame,
            )
            base_frame += cfg.frames_per_batch
            blob = pack_frme(
                n_frames=cfg.frames_per_batch,
                width=cfg.frame_width,
                height=cfg.frame_height,
                batch_id=batch_id,
                frames=frames,
            )
            await conn.send(blob)
            batch_id += 1
    except ConnectionClosed:
        # Peer disconnected (e.g. during slow 720p prefill); not a server bug.
        pass
    except asyncio.CancelledError:
        raise
    finally:
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader


def _discover_ipv4_for_remote_urls() -> list[str]:
    """Return non-loopback IPv4s to print as ``ws://`` hints (stdlib only, best-effort).

    Order: first the address chosen by the OS for a UDP "route" probe (often the
    active LAN interface), then any extra IPv4s from ``gethostbyname_ex``.
    """
    out: list[str] = []
    seen: set[str] = set()

    def add(ip: str) -> None:
        ip = ip.strip()
        if not ip or ip in seen or ip.startswith("127."):
            return
        seen.add(ip)
        out.append(ip)

    # Pick the source IPv4 the kernel would use toward the wider internet.
    for probe in ("8.8.8.8", "192.0.2.1", "198.51.100.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect((probe, 80))
                add(s.getsockname()[0])
            finally:
                s.close()
            break
        except OSError:
            continue

    for name_fn in (socket.getfqdn, socket.gethostname):
        try:
            _hn, _alias, ips = socket.gethostbyname_ex(name_fn())
            for ip in ips:
                add(ip)
        except OSError:
            continue

    return out


def _log_listen_ready(cfg: ServerConfig, server: object) -> None:
    """Print bind addresses once the asyncio listener is up (after ``serve`` enters)."""
    socks_attr = getattr(server, "sockets", None)
    socks = list(socks_attr) if socks_attr is not None else []
    print(
        f"[server] WebSocket ready — config host={cfg.host!r} port={cfg.port}",
        flush=True,
    )
    for sock in socks:
        try:
            addr = sock.getsockname()
        except OSError:
            continue
        fam = sock.family
        if fam == socket.AF_INET:
            ip, port = addr[0], addr[1]
            print(f"[server]   listen IP:port = {ip}:{port}", flush=True)
        elif fam == socket.AF_INET6:
            host, port = addr[0], addr[1]
            print(f"[server]   listen IP:port = [{host}]:{port}", flush=True)
        else:
            print(f"[server]   listen socket = {addr!r}", flush=True)
    if cfg.host in ("0.0.0.0", "", "::", "[::]"):
        print(
            f"[server]   example client URL (this host): ws://127.0.0.1:{cfg.port}",
            flush=True,
        )
        guessed = _discover_ipv4_for_remote_urls()
        if guessed:
            for ip in guessed:
                print(
                    f"[server]   remote client URL (auto): ws://{ip}:{cfg.port}",
                    flush=True,
                )
        else:
            print(
                "[server]   could not auto-detect a non-loopback IPv4 "
                "(offline or unusual network); try `hostname -I` or `ip -br a` on Linux",
                flush=True,
            )
    else:
        print(f"[server]   connect URL: ws://{cfg.host}:{cfg.port}", flush=True)


async def run_server(cfg: ServerConfig) -> None:
    """Listen until cancelled (Ctrl+C under ``asyncio.run``)."""

    async def handler(conn: ServerConnection) -> None:
        await _handle_connection(conn, cfg)

    async with serve(
        handler,
        cfg.host,
        cfg.port,
        # Pre-compressed images: disable permessage-deflate (CPU + latency).
        compression=None,
        max_size=cfg.max_ws_message_bytes,
    ) as server:
        _log_listen_ready(cfg, server)
        # Block forever until SIGINT cancels the process / event loop.
        await asyncio.Future()


def main_server(cfg: ServerConfig) -> None:
    asyncio.run(run_server(cfg))
