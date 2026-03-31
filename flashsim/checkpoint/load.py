import io
import os
from typing import Literal

from loguru import logger
import torch
from safetensors.torch import load as load_safetensors
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

try:
    from imaginaire.checkpointer.s3_filesystem import S3StorageReader
    from imaginaire.utils.easy_io import easy_io

    SUPPORT_S3 = True
except ImportError:
    SUPPORT_S3 = False

_ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH = "credentials/s3_checkpoint.secret"
_ALPADREAMS_CHECKPOINT_LOCAL_CACHE_DIR = os.path.expanduser(
    os.getenv("IMAGINAIRE_CACHE_DIR", "~/.cache/imaginaire")
)


def get_storage_reader(
    checkpoint_path: str, credential_path: str = _ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH
):
    """Get storage reader for S3 or local checkpoint.

    Args:
        checkpoint_path: The path to the checkpoint. Can be S3 or local path.

    Returns:
        The storage reader.
    """
    if checkpoint_path.startswith("s3://"):
        if SUPPORT_S3:
            return S3StorageReader(
                credential_path=credential_path, path=checkpoint_path
            )
        else:
            raise ValueError(
                "S3 support is not available. Please install imaginaire to use S3 checkpoints."
            )
    else:
        return FileSystemReader(checkpoint_path)


def load_distributed_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    check_success: bool = False,
    local_cache_dir: str = _ALPADREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH,
) -> torch.nn.Module:
    """Load distributed checkpoint into a model (Inplace).

    Args:
        model: The model to load the DCP checkpoint into.
        checkpoint_path: The path to the DCP checkpoint. Can be S3 or local path. Should be a directory path.
        check_success: Whether to check if the checkpoint is loaded successfully,
            by comparing the state dict of the model before and after loading the checkpoint.
    """
    is_s3_checkpoint = checkpoint_path.startswith("s3://")
    if is_s3_checkpoint and not SUPPORT_S3:
        raise ValueError(
            "S3 support is not available. Please install imaginaire to use S3 checkpoints."
        )

    # Set the cache checkpoint path so that next time we can just load the .pt file locally.
    local_cache_checkpoint_path = None
    if is_s3_checkpoint and local_cache_dir is not None:
        local_cache_checkpoint_path = os.path.join(
            local_cache_dir,
            checkpoint_path.split("s3://")[1].rstrip("/") + ".pt",
        )

    # Check if the local cache checkpoint path exists. If so, we load from the local cache.
    # In this case, we don't need to check for success.
    if local_cache_checkpoint_path is not None and os.path.exists(
        local_cache_checkpoint_path
    ):
        state_dict = torch.load(local_cache_checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info(
            f"Loaded successfully from the local cache: {local_cache_checkpoint_path}"
        )
        return model

    # If check_success is True, we check if the checkpoint is loaded successfully, by
    # comparing the state dict of the model before and after loading the checkpoint.
    if check_success:
        prev_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    # Load the DCP checkpoint. Note DCP load doesn't fail if there is no matching key.
    # So the best practice is to set check_success to True.
    storage_reader = get_storage_reader(
        checkpoint_path, credential_path=credential_path
    )
    state_dict = model.state_dict()
    torch.distributed.checkpoint.load(
        state_dict,
        storage_reader=storage_reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )

    # Now check if the checkpoint is loaded successfully.
    if check_success:
        for k, v in model.state_dict().items():
            prev_v = prev_state_dict[k]
            if (prev_v == v).all():
                logger.error(
                    f"DCP load seems failed for key {k}. The values are not changed!"
                )

    # Cache the state dict locally if needed..
    if local_cache_checkpoint_path is not None:
        os.makedirs(os.path.dirname(local_cache_checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), local_cache_checkpoint_path)
        logger.info(f"Loaded successfully from the checkpoint: {checkpoint_path}")
        logger.info(f"Cached locally to {local_cache_checkpoint_path}")
    else:
        logger.info(f"Loaded successfully from the checkpoint: {checkpoint_path}")

    return model


def load_single_checkpoint(
    checkpoint_path: str,
    local_cache_dir: str = _ALPADREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a single checkpoint file (.pt, .pth, .safetensors) from S3 or local.

    Supports loading from S3 with local caching for faster subsequent loads.

    Args:
        checkpoint_path: Path to the checkpoint file. Can be S3 (s3://...) or local path.
            Supported extensions: .pt, .pth, .safetensors
        local_cache_dir: Directory to cache S3 checkpoints locally.
        credential_path: Path to S3 credentials file.
        map_location: Device to map tensors to (for .pt/.pth files).

    Returns:
        State dict loaded from the checkpoint.

    Raises:
        ValueError: If the file extension is not supported.
    """
    is_s3_path = checkpoint_path.startswith("s3://")

    # Determine file extension
    ext = os.path.splitext(checkpoint_path)[1].lower()
    if ext not in (".pt", ".pth", ".safetensors"):
        raise ValueError(
            f"Unsupported checkpoint extension: {ext}. Supported: .pt, .pth, .safetensors"
        )

    # For S3 paths, check local cache first
    local_cache_path = None
    if is_s3_path and local_cache_dir is not None:
        local_cache_path = os.path.join(
            local_cache_dir, checkpoint_path.removeprefix("s3://")
        )
        if os.path.exists(local_cache_path):
            logger.info(f"Loading from local cache: {local_cache_path}")
            return _load_checkpoint_from_local(local_cache_path, ext, map_location)

    # Load from S3 or local
    if is_s3_path:
        state_dict = _load_checkpoint_from_s3(
            checkpoint_path, ext, credential_path, map_location
        )
        # Cache to local
        if local_cache_path is not None:
            os.makedirs(os.path.dirname(local_cache_path), exist_ok=True)
            _save_to_local_cache(state_dict, local_cache_path, ext)
            logger.info(f"Cached checkpoint to: {local_cache_path}")
    else:
        state_dict = _load_checkpoint_from_local(checkpoint_path, ext, map_location)

    return state_dict


def _load_checkpoint_from_local(
    path: str,
    ext: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load checkpoint from local filesystem."""
    if ext == ".safetensors":
        with open(path, "rb") as f:
            return load_safetensors(f.read())
    else:
        return torch.load(path, map_location=map_location, weights_only=False)


def _load_checkpoint_from_s3(
    s3_path: str,
    ext: str,
    credential_path: str,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load checkpoint from S3."""
    if not SUPPORT_S3:
        raise ValueError(
            "S3 support is not available. Please install imaginaire to use S3 checkpoints."
        )

    logger.info(f"Downloading checkpoint from S3: {s3_path}")
    backend_args = {"s3_credential_path": credential_path}
    data_bytes = easy_io.get(s3_path, backend_args=backend_args)

    if ext == ".safetensors":
        return load_safetensors(data_bytes)
    else:
        return torch.load(
            io.BytesIO(data_bytes), map_location=map_location, weights_only=False
        )


def _save_to_local_cache(
    state_dict: dict[str, torch.Tensor], path: str, ext: str
) -> None:
    """Save state dict to local cache."""
    from safetensors.torch import save_file as save_safetensors

    if ext == ".safetensors":
        save_safetensors(state_dict, path)
    else:
        torch.save(state_dict, path)


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module | None = None,
    checkpoint_type: Literal["auto", "single", "distributed"] = "auto",
    local_cache_dir: str = _ALPADREAMS_CHECKPOINT_LOCAL_CACHE_DIR,
    credential_path: str = _ALPADREAMS_CHECKPOINT_CREDENTIAL_PATH,
    map_location: str | torch.device = "cpu",
    check_success: bool = False,
) -> dict[str, torch.Tensor] | torch.nn.Module:
    """Unified API to load checkpoints from S3 or local filesystem.

    Supports both single file checkpoints (.pt, .pth, .safetensors) and
    distributed checkpoints (DCP format).

    Args:
        checkpoint_path: Path to checkpoint. Can be S3 (s3://...) or local.
            - For single files: path to .pt, .pth, or .safetensors file
            - For distributed: path to DCP checkpoint directory
        model: Model to load the checkpoint into. Required for distributed checkpoints.
            If provided for single checkpoints, will call model.load_state_dict().
        checkpoint_type: Type of checkpoint to load.
            - "auto": Automatically detect based on path (file vs directory)
            - "single": Force single file loading
            - "distributed": Force distributed checkpoint loading
        local_cache_dir: Directory to cache S3 checkpoints locally.
        credential_path: Path to S3 credentials file.
        map_location: Device to map tensors to (for single file checkpoints).
        check_success: For distributed checkpoints, verify loading succeeded.

    Returns:
        - If model is None: returns the state dict
        - If model is provided: returns the model with loaded weights

    Raises:
        ValueError: If checkpoint_type is "distributed" but model is not provided.
    """
    # Auto-detect checkpoint type
    if checkpoint_type == "auto":
        ext = os.path.splitext(checkpoint_path)[1].lower()
        if ext in (".pt", ".pth", ".safetensors"):
            checkpoint_type = "single"
        else:
            checkpoint_type = "distributed"

    if checkpoint_type == "single":
        state_dict = load_single_checkpoint(
            checkpoint_path=checkpoint_path,
            local_cache_dir=local_cache_dir,
            credential_path=credential_path,
            map_location=map_location,
        )
        if model is not None:
            model.load_state_dict(state_dict)
            logger.info(f"Loaded checkpoint into model: {checkpoint_path}")
            return model
        return state_dict

    elif checkpoint_type == "distributed":
        if model is None:
            raise ValueError(
                "Model must be provided for distributed checkpoint loading"
            )
        return load_distributed_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            check_success=check_success,
            local_cache_dir=local_cache_dir,
            credential_path=credential_path,
        )

    else:
        raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")
