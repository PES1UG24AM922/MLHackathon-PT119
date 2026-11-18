"""Coordinate helpers shared by EDA, feature generation, and inference."""

from __future__ import annotations

import numpy as np
from pyproj import Transformer


def get_utm_epsg(lon: float, lat: float) -> int:
    """Compute the WGS84 UTM EPSG code for a given longitude/latitude."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0.0 else 32700) + zone


def project_to_utm(
    lon: np.ndarray,
    lat: np.ndarray,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 arrays into the selected UTM metre coordinate system."""
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
