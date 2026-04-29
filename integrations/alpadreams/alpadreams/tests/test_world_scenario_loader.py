from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from alpadreams.conditioning.world_scenario.data_loaders import (
    list_loaders,
    load_scene,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
EXAMPLE_SCENE_ZIP = REPO_ROOT / "assets" / "example_data" / "alpadreams" / "clipgt.zip"


def _load_example_scene_zip_bytes() -> bytes:
    if not EXAMPLE_SCENE_ZIP.exists():
        pytest.skip(f"Missing integration-test scene archive at {EXAMPLE_SCENE_ZIP}.")
    return EXAMPLE_SCENE_ZIP.read_bytes()


def test_load_scene_direct_from_example_zip(tmp_path: Path) -> None:
    hdmap_zip_bytes = _load_example_scene_zip_bytes()

    loaders = list_loaders()
    assert "clipgt" in loaders, (
        "clipgt loader is not registered; direct scene loading cannot work. "
        f"Registered loaders: {loaders}"
    )

    extracted_scene_dir = tmp_path / "clipgt_scene"
    extracted_scene_dir.mkdir()
    with zipfile.ZipFile(io.BytesIO(hdmap_zip_bytes), "r") as zf:
        zf.extractall(extracted_scene_dir)

    scene_data = load_scene(
        extracted_scene_dir,
        camera_names=["camera_front_wide_120fov"],
        max_frames=8,
        input_pose_fps=30,
        resize_resolution_hw=(704, 1280),
    )

    assert scene_data.scene_id
    assert scene_data.num_frames > 0
    assert len(scene_data.ego_poses) == scene_data.num_frames
    assert "camera_front_wide_120fov" in scene_data.camera_models
