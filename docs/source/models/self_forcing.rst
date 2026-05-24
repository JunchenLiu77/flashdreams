.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Self-Forcing
===================================

.. raw:: html

   <div class="model-link-row">
     <a class="model-link-button" href="https://self-forcing.github.io/" target="_blank" rel="noopener noreferrer">Project page</a>
     <a class="model-link-button" href="https://arxiv.org/abs/2506.08009" target="_blank" rel="noopener noreferrer">arXiv paper</a>
     <a class="model-link-button" href="https://github.com/guandeh17/Self-Forcing" target="_blank" rel="noopener noreferrer">Official code</a>
   </div>

Self-Forcing is a Wan2.1-based text-to-video (T2V) model.
It uses a training paradigm for autoregressive video diffusion that simulates
inference-time rollout during training with KV caching, reducing the train-test
gap and enabling efficient streaming generation quality.

.. image:: https://self-forcing.github.io/static/teaser.jpg
   :alt: Self-Forcing teaser figure.
   :width: 100%

Installation
------------

.. code-block:: bash

   # from the repo root
   uv sync --project integrations/self_forcing

Running the method
------------------

To run Self-Forcing, launch one of the registered runner slugs via
``flashdreams-run``. For example:

.. code-block:: bash

   uv run --project integrations/self_forcing flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b \
       --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 7

We provide the following variants:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Method
     - Description
   * - ``self-forcing-wan2.1-t2v-1.3b``
     - Official checkpoint.
   * - ``self-forcing-wan2.1-t2v-1.3b-flash``
     - Official checkpoint. Swap WAN VAE decoder with faster TAEHV decoder.
   * - ``self-forcing-wan2.1-t2v-1.3b-anti-drift``
     - Configuration for steady long rollout (sink + sliding window, with KV cache re-ROPE).

For multi-GPU inference, use:

.. code-block:: bash

   uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b \
       --prompt "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides." \
       --pixel-height 480 --pixel-width 832 \
       --total-blocks 7

To inspect all supported CLI arguments and their default values, run:

.. code-block:: bash

   uv run --project integrations/self_forcing flashdreams-run \
       self-forcing-wan2.1-t2v-1.3b \
       --help


Profiling Benchmark
-------------------

The chart below compares total DiT runtime for FlashDreams Self-Forcing against
the `official Self-Forcing implementation <https://github.com/guandeh17/Self-Forcing>`_
and the `FastVideo implementation <https://github.com/hao-ai-lab/FastVideo>`_
under matched settings. Use it as a quick reference for expected speedup trends
across hardware and runtime choices.

.. figure:: /_static/perf/perf-0521-self-forcing.svg
   :class: benchmark-figure
   :figclass: benchmark-figure-wrap
   :alt: Self-Forcing benchmark chart.

   This chart shows the DiT total runtime (4 denoising steps) at the 6th
   autoregressive rollout on a single GPU.
   For an apples-to-apples comparison, all implementations are forced to use cuDNN attention
   backend and ``torch.compile`` for DiT network. For profiling the official implementation, see
   `this instruction <https://github.com/NVIDIA/flashdreams/tree/main/integrations/self_forcing/tests/parity_check/README.md>`_.
   For profiling the FastVideo implementation, see
   `this instruction <https://github.com/NVIDIA/flashdreams/tree/main/integrations/self_forcing/tests/baseline_fastvideo/README.md>`_.
