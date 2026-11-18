import numpy as np
import pandas as pd
import pytest

from pathsnap.decoder.integration import (
    ArtifactContractError,
    candidate_scores_from_feature_frame,
    decode_person1_person2_output,
)


class FakeEmissionModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, matrix):
        # Feature 0 directly represents a deterministic positive score here.
        p = np.clip(matrix[:, 0], 0.01, 0.99)
        return np.column_stack([1.0 - p, p])


def _artifact(**overrides):
    artifact = {
        "feature_schema_version": "features-v1",
        "model_version": "model-v1",
        "calibration_method": "isotonic",
        "feature_cols": ["distance_feature"],
        "needs_scaling": False,
        "model": FakeEmissionModel(),
    }
    artifact.update(overrides)
    return artifact


def _frames():
    gps = pd.DataFrame(
        {
            "pt_idx": [0, 1],
            "timestamp": [0.0, 1.0],
            "x_m": [1.0, 11.0],
            "y_m": [0.0, 0.0],
            "lon": [8.0, 8.0001],
            "lat": [49.0, 49.0],
        }
    )
    nodes = pd.DataFrame(
        {
            "x_m": [0.0, 10.0, 20.0],
            "y_m": [0.0, 0.0, 0.0],
            "lon": [8.0, 8.0001, 8.0002],
            "lat": [49.0, 49.0, 49.0],
        }
    )
    arcs = pd.DataFrame({"from_node_id": [0, 1], "to_node_id": [1, 2]})
    candidates = pd.DataFrame(
        {
            "trip_id": ["trip", "trip", "trip", "trip"],
            "pt_idx": [0, 0, 1, 1],
            "cand_arc_id": [0, 1, 0, 1],
            "distance_feature": [0.9, 0.1, 0.1, 0.9],
            "candidate_miss": [False, False, False, False],
        }
    )
    return gps, nodes, arcs, candidates


def test_person1_person2_adapter_decodes_label_free_frames():
    gps, nodes, arcs, candidates = _frames()
    result, metadata = decode_person1_person2_output(
        "trip", gps, nodes, arcs, candidates, _artifact(), feature_schema_version="features-v1"
    )
    assert result.decoded_arc_ids == [0, 1]
    assert metadata.calibration_method == "isotonic"


def test_adapter_requires_calibration_and_matching_feature_schema():
    _, _, _, candidates = _frames()
    with pytest.raises(ArtifactContractError, match="calibration_method"):
        candidate_scores_from_feature_frame(
            "trip", candidates, _artifact(calibration_method=""), expected_feature_schema_version="features-v1"
        )
    with pytest.raises(ArtifactContractError, match="schema mismatch"):
        candidate_scores_from_feature_frame(
            "trip", candidates, _artifact(), expected_feature_schema_version="features-v2"
        )
