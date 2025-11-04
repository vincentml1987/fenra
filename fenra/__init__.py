"""Fenra package with lazy accessors for directed memory helpers."""

from importlib import import_module
from typing import Any

_DM_EXPORTS = {
    "DirectedMemory",
    "DirectedMemoryStore",
    "directed_memories_path",
    "directed_memory_block_for_agent",
    "format_directed_memories_block",
    "get_store",
}

__all__ = sorted(_DM_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _DM_EXPORTS:
        module = import_module("directed_memory")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_DM_EXPORTS))
