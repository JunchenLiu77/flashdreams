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

"""Test client for the UltraFlashVSR gRPC service.

Usage (from repo root):
    uv run --no-sync python -m flashvsr.grpc.client --input clip.mp4 --output out.mp4

    # Use unary chunk flow instead of streaming:
    uv run --no-sync python -m flashvsr.grpc.client --input clip.mp4 --output out.mp4 --unary

    # Connect to a remote server:
    uv run --no-sync python -m flashvsr.grpc.client --server my-host:50051 --input ...

    # Send JPEG inputs and rely on the server browser viewer for output:
    uv run --no-sync python -m flashvsr.grpc.client --input clip.mp4 --input_format jpeg --display_only
"""

import argparse
import io
import sys
import time
import uuid

import grpc
import mediapy as media
import numpy as np
from flashvsr.grpc.protos import ultraflashvsr_pb2 as pb2
from flashvsr.grpc.protos import ultraflashvsr_pb2_grpc as pb2_grpc

DEFAULT_SERVER = "localhost:50051"
DEFAULT_MAX_MESSAGE_MB = 512

# Supported (first_chunk, chunk_size) pairs.
CHUNK_MODES: dict[int, tuple[int, int]] = {
    16: (13, 16),
    8: (5, 8),
}


def build_chunks(
    total_frames: int, first_chunk: int, chunk_size: int
) -> list[tuple[int, int]]:
    """Return list of (start, size) pairs for the given first/subsequent chunk sizes."""
    chunks, pos, first = [], 0, True
    while pos < total_frames:
        need = first_chunk if first else chunk_size
        size = min(need, total_frames - pos)
        if size < need:
            print(
                f"Warning: last chunk has {size} frames (need {need}). "
                f"Truncating to {pos} frames."
            )
            break
        chunks.append((pos, size))
        pos += size
        first = False
    return chunks


def video_to_rgb_bytes(frames_np: np.ndarray) -> bytes:
    """uint8 [T,H,W,3] → raw bytes."""
    return frames_np.tobytes()


def encode_jpeg_frames(frames_np: np.ndarray, quality: int) -> list[bytes]:
    """uint8 [T,H,W,3] → one JPEG byte string per frame."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "JPEG input requires Pillow in the client environment"
        ) from exc

    encoded = []
    for frame in frames_np:
        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(frame)).save(
            buf, format="JPEG", quality=quality
        )
        encoded.append(buf.getvalue())
    return encoded


def build_chunk_request(
    *,
    chunk_idx: int,
    frame_data: np.ndarray,
    scale: int,
    sparse_ratio: float,
    input_format: str,
    jpeg_quality: int,
    display_only: bool,
) -> pb2.UpscaleChunkRequest:
    T, H, W, _ = frame_data.shape
    req = pb2.UpscaleChunkRequest(
        chunk_index=chunk_idx,
        num_frames=T,
        height=H,
        width=W,
        display_only=display_only,
    )
    if input_format == "jpeg":
        req.frame_encoding = pb2.FRAME_ENCODING_JPEG
        req.frames_jpeg.extend(encode_jpeg_frames(frame_data, jpeg_quality))
    else:
        req.frame_encoding = pb2.FRAME_ENCODING_RAW_RGB
        req.frames_rgb = video_to_rgb_bytes(frame_data)
    if chunk_idx == 0:
        req.input_height = H
        req.input_width = W
        req.scale = scale
        req.sparse_ratio = sparse_ratio
    return req


def rgb_bytes_to_video(data: bytes, T: int, H: int, W: int) -> np.ndarray:
    """Raw bytes → uint8 [T,H,W,3]."""
    return np.frombuffer(data, dtype=np.uint8).reshape(T, H, W, 3)


# ---------------------------------------------------------------------------
# Streaming flow: UpscaleVideo
# ---------------------------------------------------------------------------


def upsample_stream(
    stub: pb2_grpc.UltraFlashVSRServiceStub,
    video_np: np.ndarray,
    scale: int,
    sparse_ratio: float,
    first_chunk: int,
    chunk_size: int,
    input_format: str,
    jpeg_quality: int,
    display_only: bool,
) -> np.ndarray | None:
    T, H, W, _ = video_np.shape
    chunks = build_chunks(T, first_chunk, chunk_size)
    if not chunks:
        raise ValueError(f"Video too short (need ≥ {first_chunk} frames)")

    usable = sum(s for _, s in chunks)
    if usable < T:
        video_np = video_np[:usable]
        print(f"  Using first {usable} of {T} frames.")

    def request_iter():
        for chunk_idx, (start, size) in enumerate(chunks):
            frame_data = video_np[start : start + size]  # [T, H, W, 3]
            yield build_chunk_request(
                chunk_idx=chunk_idx,
                frame_data=frame_data,
                scale=scale,
                sparse_ratio=sparse_ratio,
                input_format=input_format,
                jpeg_quality=jpeg_quality,
                display_only=display_only,
            )

    output_chunks: dict[int, np.ndarray] = {}
    t_start = time.time()
    for response in stub.UpscaleVideo(request_iter()):
        if response.error:
            raise RuntimeError(
                f"Server error on chunk {response.chunk_index}: {response.error}"
            )
        if response.frames_omitted or not response.frames_rgb:
            if not display_only:
                raise RuntimeError(
                    "Server omitted output frames; viewer mode may be enabled. "
                    "Use --display_only to skip saving an output video."
                )
        elif not display_only:
            arr = rgb_bytes_to_video(
                response.frames_rgb,
                response.num_frames,
                response.height,
                response.width,
            )
            output_chunks[response.chunk_index] = arr
        print(
            f"  Chunk {response.chunk_index + 1}/{len(chunks)}: "
            f"{response.num_frames} frames → {response.height}×{response.width}"
            f"  ({response.elapsed_ms:.0f} ms)"
        )

    total_ms = (time.time() - t_start) * 1000
    print(f"  Total: {total_ms:.0f} ms for {len(chunks)} chunks")

    if display_only:
        return None
    ordered = [output_chunks[i] for i in sorted(output_chunks)]
    return np.concatenate(ordered, axis=0)


# ---------------------------------------------------------------------------
# Unary chunk-by-chunk flow: StartSession + UpscaleChunk + EndSession
# ---------------------------------------------------------------------------


def upsample_unary(
    stub: pb2_grpc.UltraFlashVSRServiceStub,
    video_np: np.ndarray,
    scale: int,
    sparse_ratio: float,
    first_chunk: int,
    chunk_size: int,
    input_format: str,
    jpeg_quality: int,
    display_only: bool,
) -> np.ndarray | None:
    T, H, W, _ = video_np.shape
    chunks = build_chunks(T, first_chunk, chunk_size)
    if not chunks:
        raise ValueError(f"Video too short (need ≥ {first_chunk} frames)")

    usable = sum(s for _, s in chunks)
    if usable < T:
        video_np = video_np[:usable]
        print(f"  Using first {usable} of {T} frames.")

    session_id = str(uuid.uuid4())
    resp = stub.StartSession(
        pb2.StartSessionRequest(
            session_id=session_id,
            input_height=H,
            input_width=W,
            scale=scale,
            sparse_ratio=sparse_ratio,
        )
    )
    if not resp.success:
        raise RuntimeError(f"StartSession failed: {resp.error}")
    print(f"  Session: {resp.session_id}")

    output_chunks = []
    t_start = time.time()
    try:
        for chunk_idx, (start, size) in enumerate(chunks):
            frame_data = video_np[start : start + size]
            req = build_chunk_request(
                chunk_idx=chunk_idx,
                frame_data=frame_data,
                scale=scale,
                sparse_ratio=sparse_ratio,
                input_format=input_format,
                jpeg_quality=jpeg_quality,
                display_only=display_only,
            )
            req.session_id = session_id
            response = stub.UpscaleChunk(req)
            if response.error:
                raise RuntimeError(
                    f"Server error on chunk {chunk_idx}: {response.error}"
                )
            if response.frames_omitted or not response.frames_rgb:
                if not display_only:
                    raise RuntimeError(
                        "Server omitted output frames; viewer mode may be enabled. "
                        "Use --display_only to skip saving an output video."
                    )
            elif not display_only:
                arr = rgb_bytes_to_video(
                    response.frames_rgb,
                    response.num_frames,
                    response.height,
                    response.width,
                )
                output_chunks.append(arr)
            print(
                f"  Chunk {chunk_idx + 1}/{len(chunks)}: "
                f"{response.num_frames} frames → {response.height}×{response.width}"
                f"  ({response.elapsed_ms:.0f} ms)"
            )
    finally:
        stub.EndSession(pb2.EndSessionRequest(session_id=session_id))

    total_ms = (time.time() - t_start) * 1000
    print(f"  Total: {total_ms:.0f} ms for {len(chunks)} chunks")
    if display_only:
        return None
    return np.concatenate(output_chunks, axis=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="UltraFlashVSR gRPC test client")
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"host:port (default: {DEFAULT_SERVER})",
    )
    parser.add_argument("--input", required=True, help="Input video path (.mp4)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (.mp4). Required unless --display_only is set.",
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
        "--unary",
        action="store_true",
        help="Use unary UpscaleChunk flow instead of streaming",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        choices=[8, 16],
        help="Chunk size: 16 → first=13/subsequent=16, 8 → first=5/subsequent=8 (default: 16)",
    )
    parser.add_argument(
        "--input_format",
        choices=["raw", "jpeg"],
        default="raw",
        help="Client→server frame payload format (default: raw).",
    )
    parser.add_argument(
        "--input_jpeg_quality",
        type=int,
        default=90,
        help="JPEG quality when --input_format=jpeg (default: 90).",
    )
    parser.add_argument(
        "--display_only",
        action="store_true",
        help=(
            "Ask the server to omit output frames from gRPC responses. Use with "
            "server --viewer_port."
        ),
    )
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()
    if not args.display_only and not args.output:
        parser.error("--output is required unless --display_only is set")
    if not 1 <= args.input_jpeg_quality <= 100:
        parser.error("--input_jpeg_quality must be between 1 and 100")

    max_bytes = args.max_message_mb * 1024 * 1024
    channel_options = [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
    ]
    channel = grpc.insecure_channel(args.server, options=channel_options)
    stub = pb2_grpc.UltraFlashVSRServiceStub(channel)

    # Health check
    print(f"Checking server at {args.server} ...")
    try:
        status = stub.GetStatus(pb2.StatusRequest(), timeout=10)
    except grpc.RpcError as exc:
        print(f"Cannot reach server: {exc.details()}", file=sys.stderr)
        sys.exit(1)
    print(f"  ready={status.ready}  device={status.device}  model={status.model_name}")
    if not status.ready:
        print("Server not ready.", file=sys.stderr)
        sys.exit(1)

    # Read input video
    print(f"\nReading {args.input} ...")
    video_np = media.read_video(args.input)  # [T, H, W, 3] uint8
    T, H, W, _ = video_np.shape
    print(f"  {T} frames  {H}×{W}")

    fps = args.fps
    if fps is None:
        try:
            fps = media.VideoMetadata.from_path(args.input).fps
        except Exception:
            fps = 30.0
    print(f"  fps: {fps}")

    first_chunk, chunk_size = CHUNK_MODES[args.chunk_size]
    mode = "unary" if args.unary else "streaming"
    print(f"\nUpscaling ({mode}, chunks {first_chunk}/{chunk_size}) via gRPC ...")
    if args.unary:
        result = upsample_unary(
            stub,
            video_np,
            args.scale,
            args.sparse_ratio,
            first_chunk,
            chunk_size,
            args.input_format,
            args.input_jpeg_quality,
            args.display_only,
        )
    else:
        result = upsample_stream(
            stub,
            video_np,
            args.scale,
            args.sparse_ratio,
            first_chunk,
            chunk_size,
            args.input_format,
            args.input_jpeg_quality,
            args.display_only,
        )

    if args.display_only:
        print("Done. Output frames were published by the server viewer.")
        return

    assert result is not None
    out_path = args.output
    assert out_path is not None
    if not out_path.endswith(".mp4"):
        out_path += ".mp4"
    print(
        f"\nSaving {result.shape[0]} frames → {out_path}  ({result.shape[1]}×{result.shape[2]}) @ {fps} fps"
    )
    media.write_video(out_path, result, fps=fps)
    print("Done.")


if __name__ == "__main__":
    main()
