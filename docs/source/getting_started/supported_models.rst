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

Supported models
===================================

This page gives a quick orientation for first-time users. The canonical model
catalog with per-model commands and links is :doc:`/models/index`.

Autoregressive / streaming models
---------------------------------

- :doc:`Self-Forcing </models/self_forcing>`
- :doc:`Causal-Forcing </models/causal_forcing>`
- :doc:`Causal Wan2.2 </models/causal_wan22>`
- :doc:`Lingbot-World </models/lingbot_world>`
- :doc:`OmniDreams </models/omnidreams>`
- :doc:`FlashVSR </models/flashvsr>`

Bidirectional models
--------------------

- :doc:`Wan2.1 </models/wan21>`
- :doc:`Cosmos-Predict2.5 </models/cosmos_predict2>`

In FlashDreams, bidirectional models are executed through the same pipeline
interface and are treated as a single-rollout autoregressive run.

Contributing a new method
-------------------------

Want to add a new model or variant? Start from
:doc:`/developer_guides/new_recipes`.
