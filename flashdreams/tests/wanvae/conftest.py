"""Make ``impl_reference`` importable as a sibling from this directory.

``pyproject.toml`` runs pytest with ``--import-mode=importlib``, which
does not add a test file's directory to ``sys.path`` and does not build
a parent package even when ``__init__.py`` is present, so neither
``from .impl_reference import ...`` nor a bare ``from impl_reference
import ...`` works out of the box. Inject this folder into ``sys.path``
so the bare absolute form resolves.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
