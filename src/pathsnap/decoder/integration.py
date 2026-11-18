"""Adapters between the Person 1/2 artifacts and safe Person 3 decoding.

Inference stays label-free: callers supply GPS/candidate feature frames and a
model artifact.  This module deliberately has no `.route` reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import DecoderConfig
from .decoder import HMMDecoder
from .graph import RoadGraph
from .schemas import CandidateScore, DecodeResult, GPSPoint


class ArtifactContractError(ValueError):
    """Raised when Person 1/2 artifacts cannot safely enter the decoder."""


@dataclass(frozen=True)
class IntegrationMetadata:
    feature_schema_version: str
    model_version: str
    calibration_method: str


def gps_points_from_frame(gps_frame: pd.DataFrame) -> list[GPSPoint]:
    """Convert Person 1's projected GPS frame into decoder states.

    Required: `pt_idx`, `timestamp`, `x_m`, `y_m`. Optional `lon`/`lat` are
    carried into map outputs.  `usable` defaults to True.
    """
    required = {"pt_idx", "timestamp", "x_m", "y_m"}
    missing = required - set(gps_frame.columns)
    if missing:
        raise ArtifactContractError(f"GPS frame missing columns: {sorted(missing)}")
    frame = gps_frame.sort_values("pt_idx", kind="stable")
    return [
        GPSPoint(
            gps_point_idx=int(row.pt_idx),
            timestamp_s=float(row.timestamp),
            x_m=float(row.x_m),
            y_m=float(row.y_m),
            longitude=float(row.lon) if "lon" in frame.columns and pd.notna(row.lon) else None,
            latitude=float(row.lat) if "lat" in frame.columns and pd.notna(row.lat) else None,
            usable=bool(row.usable) if "usable" in frame.columns else True,
        )
        for row in frame.itertuples(index=False)
    ]


def road_graph_from_frames(nodes_frame: pd.DataFrame, arcs_frame: pd.DataFrame) -> RoadGraph:
    """Build a directed graph from Person 1's frozen coordinate/arc schema."""
    if not {"x_m", "y_m"}.issubset(nodes_frame.columns):
        raise ArtifactContractError("nodes frame must provide projected x_m and y_m")
    if not {"from_node_id", "to_node_id"}.issubset(arcs_frame.columns):
        raise ArtifactContractError("arcs frame must provide from_node_id and to_node_id")
    lonlat = None
    if {"lon", "lat"}.issubset(nodes_frame.columns):
        lonlat = nodes_frame[["lon", "lat"]].to_numpy(dtype=float)
    return RoadGraph(
        nodes_frame[["x_m", "y_m"]].to_numpy(dtype=float),
        arcs_frame[["from_node_id", "to_node_id"]].to_numpy(dtype=int),
        lonlat,
    )


def metadata_from_artifact(
    artifact: dict[str, Any], require_calibration: bool = True
) -> IntegrationMetadata:
    """Validate the version/calibration fields required for HMM emissions."""
    missing = {"feature_schema_version", "model_version"} - set(artifact)
    if missing:
        raise ArtifactContractError(f"model artifact missing metadata: {sorted(missing)}")
    calibration_method = str(artifact.get("calibration_method", ""))
    if require_calibration and not calibration_method:
        raise ArtifactContractError(
            "model artifact has no calibration_method; calibrated probabilities are required for HMM emissions"
        )
    return IntegrationMetadata(
        feature_schema_version=str(artifact["feature_schema_version"]),
        model_version=str(artifact["model_version"]),
        calibration_method=calibration_method or "uncalibrated-development-only",
    )


def candidate_scores_from_feature_frame(
    trip_id: str,
    feature_frame: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    expected_feature_schema_version: str,
    require_calibration: bool = True,
) -> tuple[list[CandidateScore], IntegrationMetadata]:
    """Score candidates with Person 2's saved model artifact.

    Contract fields intentionally make an old model fail after Person 1 changes
    alignment or feature order.  `candidate_miss` may be supplied by Person 1.
    """
    metadata = metadata_from_artifact(artifact, require_calibration=require_calibration)
    if metadata.feature_schema_version != expected_feature_schema_version:
        raise ArtifactContractError(
            "feature schema mismatch: "
            f"decoder expects {expected_feature_schema_version}, model has {metadata.feature_schema_version}"
        )
    required = {"trip_id", "pt_idx", "cand_arc_id"}
    missing = required - set(feature_frame.columns)
    if missing:
        raise ArtifactContractError(f"candidate feature frame missing columns: {sorted(missing)}")
    if not feature_frame.empty and set(feature_frame["trip_id"].astype(str)) != {str(trip_id)}:
        raise ArtifactContractError("candidate feature frame contains a different or mixed trip_id")
    feature_cols = artifact.get("feature_cols")
    model = artifact.get("model")
    if not isinstance(feature_cols, list) or model is None:
        raise ArtifactContractError("model artifact must contain feature_cols and model")
    missing_features = set(feature_cols) - set(feature_frame.columns)
    if missing_features:
        raise ArtifactContractError(f"candidate features missing: {sorted(missing_features)}")
    matrix = feature_frame[feature_cols].to_numpy(dtype=np.float32)
    scaler = artifact.get("scaler")
    if bool(artifact.get("needs_scaling")):
        if scaler is None:
            raise ArtifactContractError("model requires scaling but artifact has no scaler")
        matrix = scaler.transform(matrix)
    probabilities = _positive_class_probability(model, matrix)
    scores = []
    for row, probability in zip(feature_frame.itertuples(index=False), probabilities):
        scores.append(
            CandidateScore(
                trip_id=str(row.trip_id),
                gps_point_idx=int(row.pt_idx),
                candidate_arc_id=int(row.cand_arc_id),
                probability=float(probability),
                candidate_miss=bool(getattr(row, "candidate_miss", False)),
            )
        )
    return scores, metadata


def decode_person1_person2_output(
    trip_id: str,
    gps_frame: pd.DataFrame,
    nodes_frame: pd.DataFrame,
    arcs_frame: pd.DataFrame,
    candidate_feature_frame: pd.DataFrame,
    model_artifact: dict[str, Any],
    *,
    feature_schema_version: str,
    decoder_config: DecoderConfig | None = None,
    require_calibration: bool = True,
) -> tuple[DecodeResult, IntegrationMetadata]:
    """One label-free integration call for Person 4's FastAPI adapter."""
    scores, metadata = candidate_scores_from_feature_frame(
        trip_id,
        candidate_feature_frame,
        model_artifact,
        expected_feature_schema_version=feature_schema_version,
        require_calibration=require_calibration,
    )
    graph = road_graph_from_frames(nodes_frame, arcs_frame)
    result = HMMDecoder(graph, decoder_config).decode(
        trip_id, gps_points_from_frame(gps_frame), scores
    )
    return result, metadata


def _positive_class_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise ArtifactContractError("emission model must implement predict_proba")
    probabilities = np.asarray(model.predict_proba(matrix), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(matrix):
        raise ArtifactContractError("predict_proba returned an invalid shape")
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    matches = np.where(classes == 1)[0]
    if len(matches) != 1:
        raise ArtifactContractError("model predict_proba output has no positive class labelled 1")
    return probabilities[:, int(matches[0])]
