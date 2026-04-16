"""Optional OpenCV window to preview WebP frames from playout (laptop / local GUI)."""

from __future__ import annotations

import contextlib
from io import BytesIO

import numpy as np
from PIL import Image


def show_webp_in_window(data: bytes, *, window_name: str, window_created: list[bool]) -> None:
    """Decode WebP bytes and draw one frame. Call ``waitKey(1)`` so the window stays responsive."""
    import cv2

    im = Image.open(BytesIO(data)).convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if not window_created[0]:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created[0] = True
    cv2.imshow(window_name, bgr)
    cv2.waitKey(1)


def destroy_viewer_window() -> None:
    """Tear down HighGUI windows (call when the client session ends)."""
    with contextlib.suppress(Exception):
        import cv2

        cv2.destroyAllWindows()
