"""Resolve serializable component names identically on the driver and workers."""
from importlib import import_module


def resolve(name: str, builtins: dict):
    """Built-in names and installed 'package.module:Class' entries are interchangeable."""
    path = builtins.get(name, name)
    module, separator, symbol = path.partition(":")
    if not separator or not module or not symbol:
        raise ValueError(f"unknown component {name!r}; choose {sorted(builtins)} or package.module:Class")
    return getattr(import_module(module), symbol)
