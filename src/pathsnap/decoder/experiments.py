"""Offline experiment loading helpers; never import these in production API code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import RoadGraph
from .io import (
    read_candidate_scores_csv,
    read_gps_csv,
    read_nodes_arcs,
    read_reference_labels_csv,
)
from .schemas import CandidateScore, GPSPoint


@dataclass
class OfflineTrip:
    trip_id: str
    partition: str
    gps_points: list[GPSPoint]
    candidates: list[CandidateScore]
    graph: RoadGraph
    reference_labels: dict[int, int] | None


def load_offline_trip(entry: dict[str, Any], root: Path, require_labels: bool = False) -> OfflineTrip:
    required = {"trip_id", "partition", "gps", "nodes", "arcs", "candidates"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"manifest trip entry missing: {sorted(missing)}")
    resolve = lambda key: root / str(entry[key])
    points = read_gps_csv(resolve("gps"))
    nodes_xy, arcs, nodes_lonlat = read_nodes_arcs(resolve("nodes"), resolve("arcs"))
    labels_path = entry.get("reference_labels")
    if require_labels and not labels_path:
        raise ValueError(f"trip {entry['trip_id']} needs reference_labels for offline evaluation")
    # A manifest may retain a server-side/offline label path. Never open it
    # during prediction merely because it happens to be listed.
    labels = (
        read_reference_labels_csv(root / str(labels_path))
        if require_labels and labels_path
        else None
    )
    return OfflineTrip(
        trip_id=str(entry["trip_id"]),
        partition=str(entry["partition"]),
        gps_points=points,
        candidates=read_candidate_scores_csv(resolve("candidates")),
        graph=RoadGraph(nodes_xy, arcs, nodes_lonlat),
        reference_labels=labels,
    )
