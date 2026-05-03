---
name: python-docstring-style
description: Write Python docstrings matching the flashdreams house style — SPDX header, one-line module docstring, Google-style function docstrings (Args/Returns/Raises), PEP 257 attribute docstrings on dataclass/class fields, double-backticks for code references, imperative first sentences. Use when authoring or editing any .py file under flashdreams/, when adding a new module/class/function/field, or when the user asks about docstring style.
---

# Python docstring style (flashdreams)

House style distilled from `flashdreams/`. Match it when adding or editing Python code so agent output is indistinguishable from human-written code.

## File header

Every `.py` file starts with the SPDX + Apache-2.0 block, then a blank line, then the module docstring, then a blank line, then imports.

```python
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

"""Block KV cache for causal attention with a fixed-size local window."""
```

## Module docstring

**One line**, noun phrase (not imperative), describes what the module provides — not how.

- `"""Tensor and object splitting/gathering primitives for context parallelism."""`
- `"""Reusable CUDA-graph capture wrapper for stateful inference callables."""`
- `"""Multi-view, HDMap-conditioned Cosmos DiT for streaming alpadreams."""`

If one line genuinely doesn't fit, use a one-line summary + blank line + wrapped prose, but prefer tightening the summary.

## No stale inventories in headers (applies to every docstring)

**Headers — module / class / function summaries — describe the *role*, not the current set of internals or supported variants.** Comma-separated or parenthetical inventories of sub-modules, sub-classes, supported backends, supported model versions, supported keyword arguments, etc. become wrong the moment something is added or removed and rarely get updated. Detail belongs in the body / `Args:` / attribute docstrings where autodoc surfaces it from the actual source.

Module docstring:

- Good: `"""Text encoders for diffusion conditioning."""`
- Good: `"""Core building blocks shared across recipes."""`
- Bad: `"""Text encoders (Cosmos Qwen, UMT5)."""` (lies when a third encoder is added)
- Bad: `"""Core building blocks shared by all recipes (attention, checkpointing, distributed, I/O)."""`

Class docstring:

- Good: `"""Native attention module with configurable QKV layout and SDPA backend."""`
- Bad: `"""Native attention (math / efficient / cudnn / flash) with bhsd|bshd layout."""` (the `Literal[...]` choices in the signature are the source of truth)

Function docstring summary:

- Good: `"""Configure attention format and backend."""`
- Bad: `"""Configure attention format (bshd|bhsd) and backend (math, efficient, cudnn, flash)."""`

Exception: a deliberate *feature-scope* line that names a fixed, externally-versioned surface (e.g. `"""Unified Wan inference pipeline (Wan 2.1 / Wan 2.2, T2V and I2V)."""`) is fine — those names are stable contracts, not the package's own evolving internals.

## Function / method docstring

Google style. Imperative first sentence. Blank line before the sections. `Args:`, `Returns:`, `Raises:` — include only what applies. Use double backticks for code references (``` ``x`` ```, ``` ``None`` ```, ``` ``x.shape[seq_dim] // cp_size`` ```).

```python
def split_inputs_cp(
    x: Tensor, seq_dim: int, cp_group: ProcessGroup | None = None
) -> Tensor:
    """Slice a tensor along ``seq_dim`` to this rank's CP shard.

    Args:
        x: Input tensor.
        seq_dim: Dimension to split along (negative indexing supported).
        cp_group: CP process group; ``None`` returns ``x`` unchanged.

    Returns:
        Contiguous slice of length ``x.shape[seq_dim] // cp_size``.

    Raises:
        AssertionError: ``seq_dim`` is not divisible by the CP size.
    """
```

Rules:

- First sentence imperative (`"Slice …"`, `"Capture …"`, `"Return …"`), one line, ends with a period.
- Don't repeat the type annotation in `Args:` — only the semantic meaning and any non-obvious constraints.
- Describe `None` / default behavior inline: `"cp_group: CP process group; ``None`` returns ``x`` unchanged."`.
- `Returns:` describes shape / semantics, not the type (type is in the signature).
- Omit `Raises:` for purely internal asserts that callers shouldn't reason about; include it when a caller might want to catch / pattern-match.
- Skip docstrings entirely for trivial private helpers (`_prefix`) whose body is self-explanatory. When in doubt, write one.

## Class docstring

Short one-liner on the `"""` line when the class is self-contained:

```python
@dataclass(kw_only=True)
class CosmosTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for the Cosmos transformer."""
```

Multi-paragraph when the class has non-obvious structure, phases, or usage order. Use **custom section labels** (not Google's set) when they genuinely help — `Phases:`, `Per-step usage:`, `Note:`, `Typical usage example:`:

```python
class BlockKVCache:
    """
    KV cache for causal attention with a fixed-size local window and CUDA-graph support.

    Keys and values can have arbitrary shape ``[..., total_size, ...]``; the sequence
    (rolling) dimension is given by ``seq_dim`` (dimension index, can be negative).

    Phases:
        - Filling: cache not yet full; tokens are written contiguously;
          ``cached_k()`` / ``cached_v()`` return only the valid prefix.
        - Steady-state: cache full; each new chunk triggers a left-roll of the
          local window and overwrites the rightmost positions;
          ``cached_k()`` / ``cached_v()`` return the full buffer.

    Per-step usage:
        1. before_update(chunk_idx) — prepare (roll local window if steady-state).
        2. update(k, v) — write the new chunk's keys/values into the cache.
        3. cached_k() / cached_v() — get cached keys/values for attention.
        4. after_update(chunk_idx) — update internal bookkeeping.
    """
```

Do **not** use an `Attributes:` section in the class docstring — document attributes individually (see next section).

## Dataclass / class field docstrings

PEP 257 attribute docstrings: a triple-quoted string placed **on the line(s) after** each field declaration. Not `#` comments. Applies to both `@dataclass` fields and regular class attributes.

```python
@dataclass
class BlockKVCache:
    k_shape: tuple[int, ...]
    """Shape of the keys. Must be the same as the values shape except for the last dimension."""

    seq_dim: int
    """Sequence dimension that will be rolled. Can be negative."""

    sink_size: int = 0
    """Number of sink tokens at the start of the cache that are never evicted. Defaults to 0."""

    _prev_chunk_idx: int = -1
    """Chunk index of the last written chunk; -1 when empty."""
```

Rules:

- One sentence per attribute unless it has real nuance. Wrap at ~88 chars.
- Include the default value's meaning when it's non-obvious (`"Defaults to 0."`, `"-1 when empty."`).
- Document private (`_prefix`) fields too when their invariants matter.

## Inline comments

Comments explain *why* or a non-obvious invariant the code can't convey. Never narrate *what* the code does.

Good:

```python
# Hoist per-block KV pre-update out of the (graph-captured) network
# forward; predict_flow runs with eager_mode=False.
self.autoregressive_index = autoregressive_index
self.network_cache.before_update(autoregressive_index)
```

```python
seq_dim = x.ndim + seq_dim  # bring it to positive dimension
```

Bad (obvious narration — omit):

```python
# Increment counter
self._n_cached += self.chunk_size

# Initialize k and v
self._k = torch.empty(self.k_shape, device=self.device, dtype=self.dtype)
```

(The flashdreams code has occasional "initialize k and v" style comments — treat those as drift, not the standard; don't reproduce them in new code.)

Other conventions:

- TODO comments: `# TODO: short description` (no owner handle, no date).
- Don't annotate a change you're making — code is read later without that context. Example of what **not** to write: `# Fixed bug where X happened` or `# Changed to use Y for performance`.

## Section dividers within a file

Use `##` double-hash comments as visual dividers between logical sections. Seen throughout the codebase:

```python
## Default camera names / view-index mapping

DEFAULT_CAMERAS: tuple[str, ...] = (...)

## Per-rollout cache

@dataclass(kw_only=True)
class CosmosTransformerCache(TransformerAutoregressiveCache):
    ...
```

Place on its own line, one blank line before and after, short title. Don't use `# ---` rule lines or box-comments.

## Formatting / voice

- Line length: follow the file (most of the repo wraps around 88 chars; match what you see).
- Backticks: double backticks `` ``x`` `` inside docstrings for code, not single or triple. Single backticks in docstrings try to resolve as cross-references and will emit warnings when they can't.
- Shapes: annotate with double-backtick literals when it helps — `` ``[B, V, T, 1, H, W]`` ``. Use this liberally for tensor args.
- Voice: third person descriptive for docstring summaries of classes/attributes ("Long-lived AR cache…"); imperative for functions/methods ("Slice…", "Capture…").
- No emojis in docstrings or comments.

## Sphinx / Napoleon compatibility

Docs are built with `sphinx.ext.napoleon` (Google style) + `sphinx.ext.autodoc`, and `warningiserror = True` in `docs/source/conf.py` — **any malformed rST breaks CI**.

Practical rules:

- **Docstrings are reStructuredText, not Markdown.** Don't use `#` / `##` headings, `[text](url)` links, or triple-backtick code fences inside a docstring. For code blocks use `::` + an indented block, or the `Example:` / `Examples:` section (Napoleon renders it as a code block).
- **Cross-references use rST roles**, not double backticks, when you want a hyperlink:
  - `` :class:`CUDAGraphWrapper` ``, `` :meth:`BlockKVCache.update` ``, `` :func:`split_inputs_cp` ``, `` :attr:`BlockKVCache._k` ``, `` :mod:`flashdreams.infra.cuda_graph` ``, `` :obj:`None` ``.
  - Plain double backticks are still correct for *non-linked* literal code (argument names, shapes, small expressions).
- **Section labels Napoleon recognises** (and converts into rST admonitions / field lists):
  `Args`, `Arguments`, `Attention`, `Attributes`, `Caution`, `Danger`, `Error`, `Example`, `Examples`, `Hint`, `Important`, `Keyword Args`, `Keyword Arguments`, `Methods`, `Note`, `Notes`, `Other Parameters`, `Parameters`, `Return`, `Returns`, `Raise`, `Raises`, `References`, `See Also`, `Tip`, `Todo`, `Warning`, `Warnings`, `Warn`, `Warns`, `Yield`, `Yields`.
- **Custom section labels** like `Phases:`, `Per-step usage:`, `Typical usage example:` are **not** Napoleon-recognised. They render as plain paragraphs at best, and a stray blank-line can turn them into field-list warnings under `warningiserror`. Prefer `Note:` / `Example:` / `Examples:` (Napoleon-recognised) for anything callout-shaped. If a genuinely custom section is unavoidable, register it via `napoleon_custom_sections` in `conf.py` before using it in code.
- **`Attributes:` section is legal**, but we document fields with PEP 257 attribute docstrings (see above) and let autodoc discover them — don't duplicate.

## Anti-patterns — don't adopt

- reStructuredText `:param x:` / `:returns:` field lists — we use Google style. Napoleon tolerates mixing, but the rendered output is inconsistent.
- NumPy-style docstrings with `Parameters` / `----------` underlines — disabled via `napoleon_numpy_docstring = False`, renders wrong.
- Markdown inside docstrings (headings, fenced code blocks, hyperlink syntax) — not rST, will not render and may warn.
- Single backticks in docstrings for non-references — Sphinx treats them as unresolved cross-references under the default role and warns.
- Type info duplicated in `Args:` (`x (Tensor): …`) — the signature already has the type, and autodoc renders it.
- Boilerplate "This function does X" opener — start with the verb directly.
