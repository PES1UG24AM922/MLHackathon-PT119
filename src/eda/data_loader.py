"""Utilities for discovering and loading PathSnap trip datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TripData:
    """Container for a single PathSnap trip's raw data."""

    split: str
    trip_id: str
    trip_dir: Path
    track: pd.DataFrame  # columns: lon, lat, timestamp
    nodes: pd.DataFrame  # columns: lon, lat (0-indexed)
    arcs: pd.DataFrame   # columns: from_node, to_node (0-indexed)
    route: Optional[np.ndarray]  # 1D int array of arc_ids, or None if missing


def resolve_data_root(base_dir: Path | str = ".") -> Path:
    """Resolve the root directory containing `train/` and `student_test/`."""
    base = Path(base_dir)
    candidates = [
        base / "Student folder" / "Student folder",
        base / "Student folder",
        base,
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "student_test").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate 'train' and 'student_test' directories under {base.resolve()}"
    )


def list_trip_ids(split_dir: Path) -> list[str]:
    """Return sorted trip IDs (subdirectory names) inside a split directory."""
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def load_trip(split: str, trip_id: str, split_dir: Path) -> TripData:
    """Load `.track`, `.nodes`, `.arcs`, and `.route` for a single trip."""
    trip_dir = split_dir / trip_id

    track_path = trip_dir / f"{trip_id}.track"
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    route_path = trip_dir / f"{trip_id}.route"

    track = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={"lon": float, "lat": float, "timestamp": float},
    )
    nodes = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype={"lon": float, "lat": float},
    )
    arcs = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype={"from_node": int, "to_node": int},
    )

    route: Optional[np.ndarray] = None
    if route_path.exists() and route_path.stat().st_size > 0:
        route_df = pd.read_csv(route_path, header=None, names=["arc_id"], dtype=int)
        route = route_df["arc_id"].to_numpy(dtype=int)
    elif route_path.exists():
        route = np.array([], dtype=int)

    return TripData(
        split=split,
        trip_id=trip_id,
        trip_dir=trip_dir,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
    )
