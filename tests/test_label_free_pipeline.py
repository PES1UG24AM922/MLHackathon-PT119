from pathlib import Path

import pandas as pd

from src.eda.data_loader import TripData
from src.features.build_dataset import build_inference_features
from pathsnap.decoder.io import read_candidate_scores_csv


def test_candidate_csv_preserves_zero_padded_trip_id(tmp_path: Path):
    path = tmp_path / "scores.csv"
    pd.DataFrame(
        {
            "trip_id": ["00000000"],
            "gps_point_idx": [0],
            "candidate_arc_id": [0],
            "probability": [0.9],
        }
    ).to_csv(path, index=False)
    assert read_candidate_scores_csv(path)[0].trip_id == "00000000"


def test_label_free_candidate_and_feature_generation_needs_no_route():
    trip = TripData(
        split="evaluator_test",
        trip_id="00000000",
        trip_dir=Path("."),
        track=pd.DataFrame(
            {"lon": [8.0, 8.00005], "lat": [49.0, 49.0], "timestamp": [0.0, 1.0]}
        ),
        nodes=pd.DataFrame({"lon": [8.0, 8.0001], "lat": [49.0, 49.0]}),
        arcs=pd.DataFrame({"from_node": [0], "to_node": [1]}),
        route=None,
    )
    features, candidates = build_inference_features(trip, radius_m=100.0)
    assert not features.empty
    assert "is_true_arc" not in features
    assert "gt_arc_id" not in features
    assert candidates.gt_arc_ids is None
