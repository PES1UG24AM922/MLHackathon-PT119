"""CSV/manifest adapters for offline decoding. Inference never loads labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import CandidateScore, GPSPoint


REQUIRED_GPS_COLUMNS = {"gps_point_idx", "timestamp_s", "x_m", "y_m"}
REQUIRED_CANDIDATE_COLUMNS = {
    "trip_id",
    "gps_point_idx",
    "candidate_arc_id",
}


def read_gps_csv(path: str | Path) -> list[GPSPoint]:
    frame = pd.read_csv(path)
    missing = REQUIRED_GPS_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"GPS file missing columns: {sorted(missing)}")
    frame = frame.sort_values("gps_point_idx", kind="stable")
    points = []
    for row in frame.to_dict("records"):
        points.append(
            GPSPoint(
                gps_point_idx=int(row["gps_point_idx"]),
                timestamp_s=float(row["timestamp_s"]),
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                longitude=_optional_float(row.get("longitude")),
                latitude=_optional_float(row.get("latitude")),
                usable=bool(row.get("usable", True)),
            )
        )
    return points


def read_nodes_arcs(
    nodes_path: str | Path, arcs_path: str | Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Read Person 1's projected node and directed-arc CSV artifacts.

    Nodes require x_m/y_m. Arcs require from_node_id/to_node_id. This strict
    contract avoids silently decoding degree coordinates as metre distances.
    """
    nodes = pd.read_csv(nodes_path)
    arcs = pd.read_csv(arcs_path)
    if not {"x_m", "y_m"}.issubset(nodes.columns):
        raise ValueError("nodes CSV must contain x_m and y_m projected coordinates")
    if not {"from_node_id", "to_node_id"}.issubset(arcs.columns):
        raise ValueError("arcs CSV must contain from_node_id and to_node_id")
    lonlat = None
    if {"longitude", "latitude"}.issubset(nodes.columns):
        lonlat = nodes[["longitude", "latitude"]].to_numpy(dtype=float)
    return (
        nodes[["x_m", "y_m"]].to_numpy(dtype=float),
        arcs[["from_node_id", "to_node_id"]].to_numpy(dtype=int),
        lonlat,
    )


def read_candidate_scores_csv(path: str | Path) -> list[CandidateScore]:
    # Trip IDs are identifiers, not integers: preserve leading zeroes such as
    # `00000000` used by the supplied development partition.
    frame = pd.read_csv(path, dtype={"trip_id": str})
    missing = REQUIRED_CANDIDATE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"candidate score file missing columns: {sorted(missing)}")
    score_column = next(
        (name for name in ("calibrated_probability", "probability", "score") if name in frame.columns),
        None,
    )
    if score_column is None:
        raise ValueError("candidate score file needs calibrated_probability, probability, or score")
    return [
        CandidateScore(
            trip_id=str(row["trip_id"]),
            gps_point_idx=int(row["gps_point_idx"]),
            candidate_arc_id=int(row["candidate_arc_id"]),
            probability=float(row[score_column]),
            candidate_miss=bool(row.get("candidate_miss", False)),
            projection_ratio=_optional_float(row.get("proj_ratio")),
        )
        for row in frame.to_dict("records")
    ]


def read_reference_labels_csv(path: str | Path) -> dict[int, int]:
    """Offline-only labels. Do not call this function in production inference."""
    frame = pd.read_csv(path)
    required = {"gps_point_idx", "reference_arc_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"reference labels file missing columns: {sorted(missing)}")
    return {
        int(row["gps_point_idx"]): int(row["reference_arc_id"])
        for row in frame.to_dict("records")
        if pd.notna(row["reference_arc_id"])
    }


def read_manifest(path: str | Path) -> dict[str, object]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("trips"), list):
        raise ValueError("manifest must be a JSON object with a trips list")
    return manifest


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
