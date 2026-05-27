# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared pytest fixtures and constants for the interactive_drive test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SAMPLE_SCENE = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "scenes"
    / "clipgt-0d404ff7-2b66-498c-b047-1ed8cded60d4.usdz"
)
"""Optional real USDZ scene, downloaded by ``prepare.py``.

Tests that use this path must silently skip when the file is absent so the
suite stays green on machines/CI that haven't fetched the large asset."""

# Captured by test_app_smoke._pump_stream, printed at session end.
captured_presenter_device: str | None = None

_CI_TIER_MARKERS = {"ci_cpu", "ci_gpu", "manual"}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-assign the workspace-wide CI tier marker to every sample test.

    Root CI runs ``pytest -m ci_cpu`` and ``pytest -m ci_gpu``; the
    workspace's ``marker_enforcement`` plugin also rejects any test that
    doesn't carry exactly one of ``ci_cpu`` / ``ci_gpu`` / ``manual``.
    Rather than sprinkle ``pytestmark = pytest.mark.ci_cpu`` across 20+
    test modules, infer the right tier from the sample-local markers
    (xvfb takes precedence over gpu so a test that needs both a virtual
    display *and* a GPU still falls into the opt-in ``manual`` bucket
    -- the GPU CI runner image isn't guaranteed to have Xvfb):

    * ``xvfb`` (needs pyvirtualdisplay) -> ``manual``
    * ``gpu`` (raster backend, CUDA dispatch) -> ``ci_gpu``
    * everything else -> ``ci_cpu``

    Tests that already declare a CI tier marker explicitly are left
    alone. Running ``pytest`` from inside the sample dir keeps working
    because the auto-assigned tier markers are additive.
    """
    for item in items:
        existing = {marker.name for marker in item.iter_markers()}
        if existing & _CI_TIER_MARKERS:
            continue
        if "xvfb" in existing:
            item.add_marker(pytest.mark.manual)
        elif "gpu" in existing:
            item.add_marker(pytest.mark.ci_gpu)
        else:
            item.add_marker(pytest.mark.ci_cpu)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Print the Vulkan adapter used by smoke tests."""
    if captured_presenter_device and sys.__stderr__:
        sys.__stderr__.write(f"\n{captured_presenter_device}\n")
        sys.__stderr__.flush()
