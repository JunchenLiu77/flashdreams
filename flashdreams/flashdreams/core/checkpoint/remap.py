import re

from torch import Tensor


def remap_checkpoint_keys(
    state_dict: dict[str, Tensor], mapping: dict[str, str]
) -> dict[str, Tensor]:
    r"""Remap checkpoint keys to the new format.

    Note: if the key is not in the mapping, it will be kept as is.

    Args:
        state_dict: The state dictionary to remap.
        mapping: The mapping of old keys to new keys.

    Returns:
        The remapped state dictionary.

    Example:
        >>> mapping = {
        >>>    r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2",
        >>> }
        >>> state_dict = {
        >>>    "blocks.0.attn1.to_q.weight": torch.randn(10, 10),
        >>> }
        >>> new_state_dict = remap_checkpoint_keys(state_dict, mapping)
        >>> print(new_state_dict.keys())
        >>> Output:
        >>>     dict_keys(['blocks.0.to_q.weight'])
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        # try match the key to the mapping
        matched = False
        for old_key, new_key in mapping.items():
            if re.match(old_key, k):
                new_state_dict[re.sub(old_key, new_key, k)] = v
                matched = True
                break
        # if not matched, keep the key as is
        if not matched:
            new_state_dict[k] = v
    return new_state_dict
