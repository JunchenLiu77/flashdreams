from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


CAMERA_NAMES = [
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_front_wide_120fov",
]

DEFAULT_MOSAIC_REL_PATH = (
    "imaginaire4/data_local/gtc2026_alpamayo_cosmos_cl_demo/mv_minimal/mtyszkiewicz/26.02.26/"
    "sweep-3-14/clipgt-0300edb0-9310-4829-89f0-66743cbb8fa5/358973c6-12cf-11f1-b197-7331bb437827/"
    "videos/clipgt-clipgt-0300edb0-9310-4829-89f0-66743cbb8fa5_0_358973c6-12cf-11f1-b197-7331bb437827_mosaic.mp4/"
    "clipgt-clipgt-0300edb0-9310-4829-89f0-66743cbb8fa5_0_358973c6-12cf-11f1-b197-7331bb437827_mosaic.mp4"
)


def run_command(args: list[str]) -> None:
    subprocess.run(args, check=True)


def export_views(input_video: Path, output_dir: Path) -> None:
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required but was not found in PATH.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, camera_name in enumerate(CAMERA_NAMES):
        hdmap_output = output_dir / f"{camera_name}.mp4"
        rgb_first_frame_output = output_dir / f"{camera_name}.png"

        crop_x_expr = f"(iw/4)*{idx}"

        run_command(
            [
                ffmpeg_path,
                "-y",
                "-i",
                str(input_video),
                "-an",
                "-vf",
                f"crop=iw/4:ih/2:{crop_x_expr}:ih/2",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                str(hdmap_output),
            ]
        )

        run_command(
            [
                ffmpeg_path,
                "-y",
                "-i",
                str(input_video),
                "-vf",
                f"crop=iw/4:ih/2:{crop_x_expr}:0,select=eq(n\\,0)",
                "-vframes",
                "1",
                str(rgb_first_frame_output),
            ]
        )

        print(f"Exported: {hdmap_output.name} and {rgb_first_frame_output.name}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root.parent / DEFAULT_MOSAIC_REL_PATH
    default_output = repo_root / "assets/example_data"

    parser = argparse.ArgumentParser(
        description="Split a 2x4 mosaic video into hdmap videos and RGB first-frame images."
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        default=default_input,
        help="Path to the source 2x4 mosaic mp4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory to write exported files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_views(input_video=args.input_video, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
