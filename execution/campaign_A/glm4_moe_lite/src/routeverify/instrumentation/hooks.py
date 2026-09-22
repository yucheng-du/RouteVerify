"""Small hook utilities with explicit lifecycle management."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class HookHandleGroup:
    handles: list[Any]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


@contextmanager
def temporary_forward_hook(module: Any, hook: Callable[..., Any]) -> Iterator[None]:
    """Register a forward hook and always remove it."""

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def atomic_forward_hooks(
    registrations: Iterable[tuple[Any, Callable[..., Any]]],
) -> Iterator[HookHandleGroup]:
    """Register a hook bundle atomically and always remove every handle."""

    group = HookHandleGroup(handles=[])
    try:
        for module, hook in registrations:
            group.handles.append(module.register_forward_hook(hook))
        yield group
    finally:
        group.remove()
