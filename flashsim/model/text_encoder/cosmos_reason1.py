"""
Cosmos-Reason1 (Qwen2.5-VL based) Text Encoder used by Cosmos Predict2.

Reference:
https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/cosmos_predict2/network.py
https://huggingface.co/nvidia/Cosmos-Reason1-7B
"""

import os
from dataclasses import dataclass, field
from loguru import logger
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from flashsim.model.text_encoder.base import BaseTextEncoder
from flashsim.config import InstantiateConfig
from flashsim.model.text_encoder.utils import str2bool


@dataclass
class CosmosReason1TextEncoderConfig(InstantiateConfig["CosmosReason1TextEncoder"]):
    _target: type["CosmosReason1TextEncoder"] = field(
        default_factory=lambda: CosmosReason1TextEncoder
    )

    model_name: str = "nvidia/Cosmos-Reason1-7B"
    max_length: int = 512
    dtype: torch.dtype = torch.bfloat16
    embedding_concat_strategy: str = (
        "full_concat"  # # Checkpoint uses full_concat -> 100352 dims
    )
    n_layers_per_group: int = 5


class CosmosReason1TextEncoder(BaseTextEncoder):
    """
    Cosmos-Reason1 (Qwen2.5-VL based) Text Encoder used by Cosmos Predict2.

    Cosmos-Predict2.5 uses Cosmos-Reason1, which is based on Qwen2.5-VL-7B-Instruct.
    The text embeddings are computed using FULL_CONCAT of all 28 hidden layers,
    resulting in 100,352-dim embeddings (28 layers x 3584 hidden_size).

    The DiT model uses a crossattn_proj layer to project these to 1024 dims.

    Configuration:
        - Model: Qwen/Qwen2.5-VL-7B-Instruct
        - Hidden size: 3584
        - Num layers: 28
        - Embedding strategy: full_concat (concatenate all layers) -> 100,352 dims
        - Max sequence length: 512
    """

    # Embedding concat strategies
    FULL_CONCAT = "full_concat"  # Concatenate all layer outputs: 28 * 3584 = 100,352
    MEAN_POOLING = "mean_pooling"  # Mean of all layer outputs: 3584
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"

    def __init__(
        self,
        config: CosmosReason1TextEncoderConfig,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize Cosmos-Reason1 text encoder using Qwen2.5-VL.

        Args:
            model_name: HuggingFace model name or local path to Cosmos-Reason1.
                - HuggingFace: "nvidia/Cosmos-Reason1-7B"
                - Local path: "/path/to/Cosmos-Reason1-7B"
            max_length: Maximum sequence length for text tokenization.
            device: Device to load the model on.
            embedding_concat_strategy: How to combine layer embeddings:
                - "full_concat": Concatenate all 28 layers -> 100,352 dims
                - "mean_pooling": Mean across layers -> 3584 dims
                - "pool_every_n_layers_and_concat": Pool groups then concat
            n_layers_per_group: Number of layers per group for pool_every_n strategy.
        """

        self.max_length = config.max_length
        self.device = device
        self.dtype = config.dtype
        self.embedding_concat_strategy = config.embedding_concat_strategy
        self.n_layers_per_group = config.n_layers_per_group

        # Check if using local path
        local_files_only = os.path.isdir(config.model_name) or str2bool(
            os.getenv("LOCAL_FILES_ONLY", "false")
        )

        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            cache_dir=os.getenv("HF_HOME", None),
            local_files_only=local_files_only,
        )
        self.tokenizer = self.processor.tokenizer

        # Load model (requires transformers >= 4.49.0)
        logger.info(f"Loading Cosmos-Reason1 model from {config.model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_name,
            cache_dir=os.getenv("HF_HOME", None),
            local_files_only=local_files_only,
            dtype=config.dtype,
        )

        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        # Store model config
        self.hidden_size = self.model.config.hidden_size  # 3584 for Qwen2.5-7B
        self.num_layers = self.model.config.num_hidden_layers  # 28 for Qwen2.5-7B

    def _mean_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Mean normalize tensor by subtracting mean and dividing by std."""
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (
            tensor.std(dim=-1, keepdim=True) + 1e-8
        )

    @torch.no_grad()
    def encode(self, text: list[str]) -> torch.Tensor:
        """
        Encode text prompts to embeddings using Cosmos-Reason1 style.

        Args:
            text: Single text prompt or list of prompts

        Returns:
            Text embeddings of shape (B, max_length, embedding_dim)
        """
        assert isinstance(text, list) and len(text) > 0, "text must be a non-empty list"

        input_ids_batch = []
        for prompt in text:
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            ]

            formatted = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                return_tensors="pt",
            )

            # Pad or truncate to max_length
            if formatted.shape[1] < self.max_length:
                pad_len = self.max_length - formatted.shape[1]
                padding = torch.full(
                    (1, pad_len),
                    self.tokenizer.pad_token_id or 0,
                    dtype=formatted.dtype,
                )
                formatted = torch.cat([formatted, padding], dim=1)
            else:
                formatted = formatted[:, : self.max_length]
            input_ids_batch.append(formatted)

        input_ids = torch.cat(input_ids_batch, dim=0).to(self.device)

        # Forward pass with hidden states
        outputs = self.model(
            input_ids=input_ids, output_hidden_states=True, return_dict=True
        )
        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors

        # Skip embedding layer (index 0), normalize and combine layers
        normalized_hidden_states = [
            self._mean_normalize(hidden_states[i]) for i in range(1, len(hidden_states))
        ]

        if self.embedding_concat_strategy == self.FULL_CONCAT:
            text_embeddings = torch.cat(normalized_hidden_states, dim=-1)
        elif self.embedding_concat_strategy == self.MEAN_POOLING:
            text_embeddings = torch.stack(normalized_hidden_states).mean(dim=0)
        elif self.embedding_concat_strategy == self.POOL_EVERY_N_LAYERS_AND_CONCAT:
            pooled = []
            for i in range(0, len(normalized_hidden_states), self.n_layers_per_group):
                group = normalized_hidden_states[i : i + self.n_layers_per_group]
                pooled.append(torch.stack(group).mean(dim=0))
            text_embeddings = torch.cat(pooled, dim=-1)
        else:
            raise ValueError(
                f"Invalid embedding_concat_strategy: {self.embedding_concat_strategy}"
            )

        return text_embeddings

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension based on concat strategy."""
        if self.embedding_concat_strategy == self.FULL_CONCAT:
            return self.num_layers * self.hidden_size
        elif self.embedding_concat_strategy == self.MEAN_POOLING:
            return self.hidden_size
        elif self.embedding_concat_strategy == self.POOL_EVERY_N_LAYERS_AND_CONCAT:
            n_groups = (
                self.num_layers + self.n_layers_per_group - 1
            ) // self.n_layers_per_group
            return n_groups * self.hidden_size
        return self.hidden_size


if __name__ == "__main__":
    text_encoder = CosmosReason1TextEncoder(config=CosmosReason1TextEncoderConfig())

    text = ["A beautiful sunset over a calm ocean."]
    text_embeddings = text_encoder.encode(text)

    # TODO: this is slightly different from I4 projects/cosmos/sil/_alpadreams/inference/text_encoder.py
    print(
        f"text_embeddings.shape: {text_embeddings.shape}"
    )  # torch.Size([1, 512, 100352])
    print(f"text_embeddings.dtype: {text_embeddings.dtype}")  # torch.bfloat16
    print(f"text_embeddings.device: {text_embeddings.device}")  # cuda:0
    print(f"text_embeddings.sum: {text_embeddings.sum()}")  # 148.0
    print(
        f"text_embeddings.flatten()[:5]: {text_embeddings.flatten()[:5]}"
    )  # [-0.7969, -0.0889,  0.1748, -0.0898,  0.0361]
    print(
        f"text_embeddings.flatten()[-5:]: {text_embeddings.flatten()[-5:]}"
    )  # [ 1.5312,  0.1309,  0.5039, -0.2354,  0.1465]
