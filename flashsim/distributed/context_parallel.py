from typing import Any

import torch
from torch import Tensor
from torch.distributed import (
    ProcessGroup,
    all_gather,
    all_gather_object,
    get_world_size,
)


def split_inputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Split input tensor along the sequence dimension for context parallelism.

    This function divides the input tensor into equal parts along the specified
    sequence dimension, based on the number of ranks in the context parallelism group.
    It then selects the part corresponding to the current rank.

    Args:
        x: Input tensor to be split.
        seq_dim: The dimension along which to split the input (sequence dimension).
        cp_group: The process group for context parallelism.

    Returns:
        A slice of the input tensor corresponding to the current rank.

    Raises:
        AssertionError: If the sequence dimension is not divisible by the number of ranks.
    """
    cp_size = cp_group.size()

    assert x.shape[seq_dim] % cp_size == 0, (
        f"{x.shape[seq_dim]} cannot divide cp_size {cp_size}"
    )
    x = x.view(
        *x.shape[:seq_dim],
        cp_size,
        x.shape[seq_dim] // cp_size,
        *x.shape[(seq_dim + 1) :],
    )
    seq_idx = torch.tensor([cp_group.rank()], device=x.device)
    x = x.index_select(seq_dim, seq_idx)
    # Note that the new sequence length is the original sequence length / cp_size
    x = x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :])
    return x.contiguous()


def cat_outputs_cp(x: Tensor, seq_dim: int, cp_group: ProcessGroup) -> Tensor:
    """
    Concatenate outputs from different ranks in the checkpoint parallelism group.

    This function gathers tensors from all ranks in the checkpoint parallelism group
    and concatenates them along the specified sequence dimension.

    Args:
        x: Input tensor to be concatenated.
        seq_dim: The dimension along which to concatenate the tensors (sequence dimension).
        cp_group: The process group for checkpoint parallelism.

    Returns:
        A tensor that is the concatenation of tensors from all ranks in the cp_group.

    Raises:
        RuntimeError: If the gather operation fails.
    """
    x = x.contiguous()

    # Get the world size (number of processes in the group)
    world_size = get_world_size(cp_group)

    # Create a list to store tensors from all ranks
    gathered_tensors = [torch.zeros_like(x) for _ in range(world_size)]

    # Gather tensors from all ranks
    try:
        all_gather(gathered_tensors, x, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError("Failed to gather tensors") from e

    # Concatenate the gathered tensors along the specified dimension
    return torch.cat(gathered_tensors, dim=seq_dim)


def split_inputs_cp_object_list(
    object_list: list[Any], cp_group: ProcessGroup
) -> list[Any]:
    """
    Split input object list for context parallelism.

    This function divides the input object list into equal parts, based on the number of ranks in the context parallelism group.
    It then selects the part corresponding to the current rank.

    Args:
        object_list: List of objects to be split.
        cp_group: The process group for context parallelism.

    Returns:
        A list of objects corresponding to the current rank.

    Raises:
        AssertionError: If the sequence dimension is not divisible by the number of ranks.
    """
    cp_size = cp_group.size()
    n_objects = len(object_list)
    assert n_objects % cp_size == 0, f"{n_objects} cannot divide cp_size {cp_size}"

    n_objects_per_rank = n_objects // cp_size
    rank = cp_group.rank()
    start_idx = rank * n_objects_per_rank
    end_idx = start_idx + n_objects_per_rank
    return object_list[start_idx:end_idx]


def cat_outputs_cp_object_list(
    object_list: list[Any], cp_group: ProcessGroup
) -> list[Any]:
    """
    Concatenate outputs from different ranks in the context parallelism group.

    This function gathers objects from all ranks in the context parallelism group
    and concatenates them into a single list.

    Args:
        object_list: List of objects to be gathered on current rank.
        cp_group: The process group for context parallelism.

    Returns:
        A list of objects that is the concatenation of objects from all ranks in the cp_group.
    """
    # Get the world size (number of processes in the group)
    world_size = get_world_size(cp_group)

    # Create a list to store tensors from all ranks
    gathered_object_list: list[list[Any] | None] = [None for _ in range(world_size)]

    # Gather tensors from all ranks
    try:
        all_gather_object(gathered_object_list, object_list, group=cp_group)
    except RuntimeError as e:
        raise RuntimeError("Failed to gather tensors") from e

    # `all_gather_object` is treating `object_list` as a single object.
    # since we are passing in a list, the resulted `gathered_object_list` would be
    # a list of lists. We need to flatten the list of lists.
    return [item for sublist in gathered_object_list for item in sublist]
