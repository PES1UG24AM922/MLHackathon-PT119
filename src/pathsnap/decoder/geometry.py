"""Small dependency-free geometry helpers used by candidate and decoder code."""

from __future__ import annotations

import math


def euclidean_m(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def bearing_radians(x1: float, y1: float, x2: float, y2: float) -> float | None:
    """Return a projected-coordinate bearing, or None for a zero-length vector."""
    if x1 == x2 and y1 == y2:
        return None
    return math.atan2(y2 - y1, x2 - x1)


def angular_difference_radians(a: float, b: float) -> float:
    """Smallest absolute angular difference in [0, pi]."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)
