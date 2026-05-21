#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Continuously stream one video to UltraFlashVSR in 8-frame chunks.

This is a live-ingest stress client for the browser-viewer server path:

    uv run --no-sync python -m flashvsr.grpc.continuous_stream_client \
        --server localhost:50051 \
        --input /path/to/clip.mp4

By default it loops forever and sends one 8-frame request every 8 / 30 seconds,
which models production ingest. On this machine FlashVSR is expected to run
closer to 7 fps, so a long run should exercise server buffering and gRPC
backpressure. Use ``--max_chunks`` for a finite smoke test.
"""

import argparse
import sys
import time
import uuid
from collections import deque
from collections.abc import Iterator

import grpc
import mediapy as media
import numpy as np
from flashvsr.grpc.client import build_chunk_request
from flashvsr.grpc.protos import ultraflashvsr_pb2 as pb2
from flashvsr.grpc.protos import ultraflashvsr_pb2_grpc as pb2_grpc

DEFAULT_SERVER = "localhost:50051"
DEFAULT_MAX_MESSAGE_MB = 512
CHUNK_FRAMES = 8


def _read_video(path: str) -> np.ndarray:
    frames = media.read_video(path)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected RGB video [T,H,W,3], got shape {frames.shape}")
    if frames.dtype == np.uint8:
        return np.ascontiguousarray(frames)
    if np.issubdtype(frames.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(frames)) <= 1.0 else 1.0
        frames = np.clip(frames * scale, 0, 255)
    return np.ascontiguousarray(frames.astype(np.uint8))


def _circular_chunk(frames: np.ndarray, start: int, size: int) -> np.ndarray:
    total = int(frames.shape[0])
    if total <= 0:
        raise ValueError("input video has no frames")
    indexes = (np.arange(size) + start) % total
    return np.ascontiguousarray(frames[indexes])


def _video_fps(path: str) -> float:
    try:
        return float(media.VideoMetadata.from_path(path).fps)
    except Exception:
        return 30.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Endless 8-frame UltraFlashVSR streaming stress client"
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"host:port (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input video to loop continuously.",
    )
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument(
        "--sparse_ratio",
        type=float,
        default=0.0,
        help=(
            "Block-sparse attention ratio. Use 0 to use the server default; "
            "1.5 is faster, 2.0 is more stable."
        ),
    )
    parser.add_argument("--max_message_mb", type=int, default=DEFAULT_MAX_MESSAGE_MB)
    parser.add_argument(
        "--target_fps",
        type=float,
        default=30.0,
        help=(
            "Ingress rate to simulate. Default 30 fps models production; "
            "local FlashVSR may only consume around 7 fps."
        ),
    )
    parser.add_argument(
        "--no_pace",
        action="store_true",
        help="Send as fast as gRPC backpressure allows instead of pacing.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help="Stop after this many 8-frame chunks. Default 0 means endless.",
    )
    parser.add_argument(
        "--input_format",
        choices=["raw", "jpeg"],
        default="jpeg",
        help="Client→server frame payload format (default: jpeg).",
    )
    parser.add_argument(
        "--input_jpeg_quality",
        type=int,
        default=90,
        help="JPEG quality when --input_format=jpeg (default: 90).",
    )
    parser.add_argument(
        "--return_frames",
        action="store_true",
        help=(
            "Ask the server to return raw upsampled frames over gRPC. By default "
            "responses are metadata-only and frames go to the server viewer."
        ),
    )
    parser.add_argument(
        "--report_every",
        type=int,
        default=10,
        help="Print progress every N sent/received chunks (default: 10).",
    )
    parser.add_argument(
        "--ingress_window_chunks",
        type=int,
        default=16,
        help=(
            "Number of recent sent chunks used for observed ingress FPS "
            "(default: 16)."
        ),
    )
    args = parser.parse_args()

    if args.target_fps <= 0:
        parser.error("--target_fps must be positive")
    if args.max_chunks < 0:
        parser.error("--max_chunks must be non-negative")
    if args.report_every <= 0:
        parser.error("--report_every must be positive")
    if args.ingress_window_chunks < 2:
        parser.error("--ingress_window_chunks must be at least 2")
    if not 1 <= args.input_jpeg_quality <= 100:
        parser.error("--input_jpeg_quality must be between 1 and 100")

    frames = _read_video(args.input)
    total_frames, height, width, _ = frames.shape
    source_fps = _video_fps(args.input)
    print(
        f"Loaded {args.input}: {total_frames} frames, {width}x{height}, "
        f"source_fps={source_fps:.2f}"
    )
    print(
        f"Streaming {CHUNK_FRAMES}-frame chunks to {args.server}; "
        f"target_ingress={args.target_fps:.2f} fps; "
        f"mode={'return_frames' if args.return_frames else 'display_only'}"
    )

    max_bytes = args.max_message_mb * 1024 * 1024
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ],
    )
    stub = pb2_grpc.UltraFlashVSRServiceStub(channel)

    try:
        status = stub.GetStatus(pb2.StatusRequest(), timeout=10)
    except grpc.RpcError as exc:
        print(f"Cannot reach server: {exc.details()}", file=sys.stderr)
        sys.exit(1)
    if not status.ready:
        print("Server is not ready.", file=sys.stderr)
        sys.exit(1)
    print(f"Server ready: device={status.device} model={status.model_name}")

    session_id = str(uuid.uuid4())
    state = {"sent": 0, "received": 0}
    send_times: deque[float] = deque(maxlen=args.ingress_window_chunks)
    receive_times: deque[float] = deque(maxlen=args.ingress_window_chunks)
    frame_interval = CHUNK_FRAMES / args.target_fps

    def requests() -> Iterator[pb2.UpscaleChunkRequest]:
        next_send_at = time.perf_counter()
        source_pos = 0
        chunk_idx = 0
        while args.max_chunks == 0 or chunk_idx < args.max_chunks:
            if not args.no_pace:
                now = time.perf_counter()
                if now < next_send_at:
                    time.sleep(next_send_at - now)
                else:
                    next_send_at = now
                next_send_at += frame_interval

            chunk = _circular_chunk(frames, source_pos, CHUNK_FRAMES)
            req = build_chunk_request(
                chunk_idx=chunk_idx,
                frame_data=chunk,
                scale=args.scale,
                sparse_ratio=args.sparse_ratio,
                input_format=args.input_format,
                jpeg_quality=args.input_jpeg_quality,
                display_only=not args.return_frames,
            )
            req.session_id = session_id

            send_times.append(time.perf_counter())
            state["sent"] += 1
            source_pos = (source_pos + CHUNK_FRAMES) % total_frames
            chunk_idx += 1
            if state["sent"] % args.report_every == 0:
                elapsed = max(send_times[-1] - send_times[0], 1e-6)
                ingress_fps = (len(send_times) - 1) * CHUNK_FRAMES / elapsed
                print(
                    f"sent_chunks={state['sent']} sent_frames={state['sent'] * CHUNK_FRAMES} "
                    f"observed_ingress={ingress_fps:.2f} fps "
                    f"window_chunks={len(send_times)}"
                )
            yield req

    try:
        for response in stub.UpscaleVideo(requests()):
            if response.error:
                raise RuntimeError(
                    f"server error on chunk {response.chunk_index}: {response.error}"
                )
            state["received"] += 1
            if args.return_frames and not response.frames_rgb:
                raise RuntimeError(
                    f"chunk {response.chunk_index} returned no frames; "
                    "server may be in viewer-only mode"
                )
            receive_times.append(time.perf_counter())
            if state["received"] % args.report_every == 0:
                elapsed = max(receive_times[-1] - receive_times[0], 1e-6)
                receive_fps = (
                    (len(receive_times) - 1) * CHUNK_FRAMES / elapsed
                )
                print(
                    f"received_chunks={state['received']} "
                    f"last_chunk={response.chunk_index} "
                    f"last_elapsed_ms={response.elapsed_ms:.0f} "
                    f"frames_omitted={response.frames_omitted} "
                    f"observed_receive={receive_fps:.2f} fps "
                    f"window_chunks={len(receive_times)}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted; closing stream.")
    finally:
        channel.close()

    print(
        f"Done. sent_chunks={state['sent']} received_chunks={state['received']} "
        f"session={session_id}"
    )


if __name__ == "__main__":
    main()
