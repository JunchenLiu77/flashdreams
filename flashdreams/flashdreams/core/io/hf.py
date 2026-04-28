"""Small Hugging Face helpers shared across encoders.

Exposes :func:`should_use_local_files_only`, which returns ``True`` when
``transformers.from_pretrained`` should be called with
``local_files_only=True``. That flag is the only way to suppress the
HTTP ``model_info`` API call that ``transformers`` does on every
``tokenizer.from_pretrained`` (e.g. inside ``_patch_mistral_regex``);
without it, multi-rank loads of the same repo trip HF's per-IP 429
limiter even when the repo is fully cached on disk.
"""

from __future__ import annotations

import os


def _str2bool(v: str | bool) -> bool:
    """Parse the usual yes/no/true/false/1/0 strings into a bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError(f"Boolean value expected, got {v!r}")


def _hf_repo_is_cached(repo_id: str) -> bool:
    """Whether ``repo_id`` is already present in the standard HF hub cache.

    Probes a handful of common root-level files via
    ``huggingface_hub.try_to_load_from_cache``. We try multiple names because
    the file at a repo root depends on the repo type:

    - ``config.json`` for plain ``transformers`` model repos.
    - ``model_index.json`` for Diffusers pipeline repos
      (e.g. ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers``), which only have
      ``config.json`` inside subfolders.
    - ``tokenizer_config.json`` / ``preprocessor_config.json`` for repos
      that ship just a tokenizer or processor.

    If any one of these is present locally, every other asset that was
    downloaded alongside it (weights, tokenizer, processor, subfolders)
    will also load with ``local_files_only=True``.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False

    candidates = (
        "config.json",
        "model_index.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    )
    for filename in candidates:
        try:
            if try_to_load_from_cache(repo_id, filename) is not None:
                return True
        except Exception:
            continue
    return False


def should_use_local_files_only(repo_id_or_path: str) -> bool:
    """Decide whether to pass ``local_files_only=True`` to ``from_pretrained``.

    Returns ``True`` if any of the following hold:

    - ``repo_id_or_path`` is a local directory.
    - The standard ``HF_HUB_OFFLINE`` env var is truthy.
    - The legacy ``LOCAL_FILES_ONLY`` env var is truthy.
    - The standard HF hub cache already contains ``repo_id_or_path``.
    """
    return (
        os.path.isdir(repo_id_or_path)
        or _str2bool(os.getenv("HF_HUB_OFFLINE", "false"))
        or _str2bool(os.getenv("LOCAL_FILES_ONLY", "false"))
        or _hf_repo_is_cached(repo_id_or_path)
    )
