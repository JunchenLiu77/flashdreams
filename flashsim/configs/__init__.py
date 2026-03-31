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
        return self._target(self, **kwargs)
