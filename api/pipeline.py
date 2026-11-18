"""Translate shared candidate features into the Person 3 decoder contract."""
import hashlib
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import sklearn
from pydantic import BaseModel, ConfigDict, Field

from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.decoder import HMMDecoder
from pathsnap.decoder.graph import RoadGraph
from pathsnap.decoder.schemas import CandidateScore, GPSPoint
from src.features.build_dataset import build_inference_features
from src.features.feature_engineering import FEATURE_COLS


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"]
    feature_version: Literal["pathsnap-features-v1", "pathsnap-features-v2-ordered-labels"]
    dataset_version: str
    model_version: str
    decoder_version: str
    random_seed: int
    sklearn_version: str
    model_file: Path
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoder_config: Path
    decoder_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score_kind: Literal["uncalibrated", "calibrated"]
    radius_m: float = Field(gt=0)
    max_candidates: int = Field(ge=1)


def verified_path(root, path, checksum, *, text=False):
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Artifact path is outside the repository")
    content = resolved.read_bytes()
    if text:
        content = content.replace(b"\r\n", b"\n")
    if hashlib.sha256(content).hexdigest() != checksum:
        raise ValueError("Artifact checksum mismatch")
    return resolved


class TeamPipeline:
    def __init__(self, root: Path, manifest_path: Path):
        self.manifest = Manifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest = self.manifest
        if sklearn.__version__ != manifest.sklearn_version:
            raise ValueError(f"Install scikit-learn=={manifest.sklearn_version} for this model")
        model_file = verified_path(root, manifest.model_file, manifest.model_sha256)
        decoder_file = verified_path(root, manifest.decoder_config, manifest.decoder_sha256, text=True)
        # This is the team's committed artifact, pinned by checksum in the manifest.
        bundle = joblib.load(model_file)
        if manifest.feature_version != "pathsnap-features-v1":
            if (bundle.get("feature_schema_version") != manifest.feature_version
                    or bundle.get("model_version") != manifest.model_version):
                raise ValueError("Model bundle does not match the feature and model manifest")
            if manifest.score_kind == "calibrated" and not bundle.get("calibration_method"):
                raise ValueError("Calibrated model has no calibration metadata")
        self.model = bundle["model"]
        self.scaler = bundle["scaler"] if bundle["needs_scaling"] else None
        if list(bundle["feature_cols"]) != FEATURE_COLS or 1 not in self.model.classes_:
            raise ValueError("The model's feature or probability schema is incompatible")
        if bundle["needs_scaling"] and self.scaler is None:
            raise ValueError("The model requires a fitted scaler")
        self.positive_class = list(self.model.classes_).index(1)
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.decoder_config = DecoderConfig.from_file(decoder_file)
        if self.decoder_config.decoder_version != manifest.decoder_version:
            raise ValueError("Decoder version mismatch")

    def predict_trip(self, trip):
        if trip.route is not None:
            raise ValueError("Prediction requires an unlabeled trip")
        frame, candidates = build_inference_features(
            trip, radius_m=self.manifest.radius_m, max_candidates=self.manifest.max_candidates
        )
        features = frame[FEATURE_COLS].to_numpy(dtype=np.float32)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features)[:, self.positive_class]
        graph = RoadGraph(
            np.column_stack((candidates.node_x, candidates.node_y)),
            np.column_stack((candidates.u, candidates.v)),
            trip.nodes[["lon", "lat"]].to_numpy(),
        )
        gps = [
            GPSPoint(idx, float(ts), float(x), float(y), float(lon), float(lat))
            for idx, (ts, x, y, lon, lat) in enumerate(zip(
                candidates.ts, candidates.trk_x, candidates.trk_y, trip.track.lon, trip.track.lat
            ))
        ]
        scores = [
            CandidateScore(trip.trip_id, int(idx), int(arc), float(prob), projection_ratio=float(ratio))
            for idx, arc, prob, ratio in zip(
                candidates.pt_idx, candidates.cand_arc_id, probabilities, candidates.proj_ratio
            )
        ]
        decoded = HMMDecoder(graph, self.decoder_config).decode(trip.trip_id, gps, scores)
        segments, arc_segments, disconnected = display_route(
            decoded.decoded_points, graph, self.decoder_config.max_network_search_m
        )
        warnings = {warning.code for warning in decoded.warnings}
        warnings.discard("INITIAL_SEGMENT")
        if disconnected:
            warnings.add("DISPLAY_ROUTE_DISCONNECTED")
        if candidates.n_zero_cands:
            warnings.add("CANDIDATE_RADIUS_FALLBACK")
        return decoded.decoded_arc_ids, segments, arc_segments, sorted(warnings)


def display_route(points, graph, max_distance_m):
    """Expand only within a decoder segment and break at unreachable transitions."""
    segments = []
    arc_segments = []
    disconnected = False
    previous = None
    for point in points:
        arc = point.decoded_arc_id
        if arc is None:
            previous = None
            continue
        if previous is None or previous.segment_id != point.segment_id:
            arc_segments.append([arc])
        elif previous.decoded_arc_id != arc:
            path = graph.arc_transition_path(previous.decoded_arc_id, arc, max_distance_m)
            if path is None:
                disconnected = True
                arc_segments.append([arc])
            else:
                arc_segments[-1].extend(path[1][1:])
        previous = point
    for segment in arc_segments:
        start = graph.arc_geometry_latlon(segment[0])[0]
        endpoints = [graph.arc_geometry_latlon(arc)[1] for arc in segment]
        segments.append([{"lon": lon, "lat": lat} for lat, lon in [start, *endpoints]])
    return segments, arc_segments, disconnected
