import copy
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


# Pretty printing class
class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


T = TypeVar("T")


# Base instantiate configs
@dataclass
class InstantiateConfig(Generic[T], PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: type[T]

    def setup(self, **kwargs: Any) -> T:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)  # type: ignore[call-arg]


def derive_config(
    base_config: InstantiateConfig[T], **changes: Any
) -> InstantiateConfig[T]:
    """
    Derive a new config from a base config by applying changes.

    Example:
        >>> base_config = AlpadreamsPipelineConfig(
        >>>     tokenizer=WanVAEInterfaceConfig(
        >>>         checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        >>>     ),
        >>>     detokenizer=WanVAEInterfaceConfig(
        >>>         checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["vae"],
        >>>     ),
        >>> )

        >>> new_config = derive_config(
        >>>     base_config,
        >>>     tokenizer=WanVAEInterfaceConfig(
        >>>         checkpoint_path=AVAILABLE_WAN_VAE_CHECKPOINT_PATHS["lightvae"],
        >>>     ),
        >>>     dit=dict(
        >>>         len_t=3,
        >>>         checkpoint_path=AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["single_view"]["vae_encoding"]["chunk3"],
        >>>     )
        >>> )
    """

    def _is_patchable_object(x: Any) -> bool:
        # Object is patchable if it has attribute storage (__dict__).
        return hasattr(x, "__dict__")

    def _get_field(target: Any, key: str, path: str) -> Any:
        if isinstance(target, dict):
            if key not in target:
                raise KeyError(f"Unknown key at {path}: {key}")
            return target[key]
        if hasattr(target, key):
            return getattr(target, key)
        raise KeyError(f"Unknown field at {path}: {type(target).__name__}.{key}")

    def _set_field(target: Any, key: str, value: Any, path: str) -> None:
        if isinstance(target, dict):
            if key not in target:
                raise KeyError(f"Unknown key at {path}: {key}")
            target[key] = value
            return
        if hasattr(target, key):
            setattr(target, key, value)
            return
        raise KeyError(f"Unknown field at {path}: {type(target).__name__}.{key}")

    def _recursive_patch(
        target: Any, patch: dict[str, Any], path: str = "root"
    ) -> None:
        # Apply patch recursively to dict/object target.
        for key, value in patch.items():
            current = _get_field(target, key, path)
            current_path = f"{path}.{key}"
            if isinstance(value, dict):
                # Nested patch: current can be dict or object.
                if isinstance(current, dict) or _is_patchable_object(current):
                    _recursive_patch(current, value, current_path)
                else:
                    raise TypeError(
                        f"Cannot apply nested dict patch to non-nested field at {current_path} "
                        f"(type={type(current).__name__})"
                    )
            else:
                # Leaf patch: direct assignment.
                _set_field(target, key, value, path)

    # Deep-copy base config so original config is never mutated.
    cfg = copy.deepcopy(base_config)
    _recursive_patch(cfg, changes, path=type(cfg).__name__)
    return cfg
