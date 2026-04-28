"""Make ``impl_reference_*`` modules importable as siblings.

``pyproject.toml`` runs pytest with ``--import-mode=importlib``, which
does not add a test file's directory to ``sys.path``, so neither
``from .impl_reference_flow_match import ...`` nor a bare
``from impl_reference_flow_match import ...`` works out of the box.
Inject this folder into ``sys.path`` so the bare absolute form resolves.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
