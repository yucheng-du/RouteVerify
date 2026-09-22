"""Capture helpers used by adapter implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ArrayCapture:
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, name: str, value: Any) -> None:
        array = np.asarray(value)
        self.arrays[name] = array.copy()

    def assert_unchanged(self, before: np.ndarray, after: np.ndarray, *, atol: float = 1e-6) -> None:
        if before.shape != after.shape:
            raise AssertionError(f"shape changed: before={before.shape} after={after.shape}")
        if not np.allclose(before, after, atol=atol, rtol=0):
            raise AssertionError("capture changed model output beyond tolerance")


def select_token_positions(array: np.ndarray, positions: tuple[int, ...]) -> np.ndarray:
    """Select sequence positions from an activation array."""

    if array.ndim < 2:
        raise ValueError("activation array must include sequence dimension")
    return np.take(array, indices=list(positions), axis=1)
