import numpy as np

from pathsnap.decoder.evaluation import evaluate_predictions, final_candidate_miss_rate, macro_f1
from pathsnap.decoder.graph import RoadGraph
from pathsnap.decoder.schemas import CandidateScore, GPSPoint


def test_metrics_use_point_level_predictions():
    graph = RoadGraph(
        np.array([[0, 0], [10, 0], [20, 0]], dtype=float), np.array([[0, 1], [1, 2]], dtype=int)
    )
    points = [GPSPoint(0, 0, 0, 0), GPSPoint(1, 1, 10, 0)]
    metrics = evaluate_predictions({0: 0, 1: 1}, {0: 0, 1: 1}, graph, points)
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["route_continuity"] == 1.0


def test_macro_f1_handles_missed_class():
    assert macro_f1([0, 1], [0, 0]) == 1 / 3


def test_unmatched_eligible_point_counts_against_accuracy_and_coverage():
    graph = RoadGraph(
        np.array([[0, 0], [10, 0]], dtype=float), np.array([[0, 1]], dtype=int)
    )
    points = [GPSPoint(0, 0, 0, 0), GPSPoint(1, 1, 10, 0)]
    metrics = evaluate_predictions({0: 0, 1: None}, {0: 0, 1: 0}, graph, points)
    assert metrics["top1_accuracy"] == 0.5
    assert metrics["prediction_coverage"] == 0.5
    assert metrics["n_unmatched"] == 1


def test_candidate_miss_rate_uses_final_arc_membership_including_empty_sets():
    points = [GPSPoint(index, index, 0, 0, usable=index != 4) for index in range(6)]
    candidates = [
        CandidateScore("trip", 0, 7, 1.0),  # Wrong arc, despite the default false miss flag.
        CandidateScore("trip", 1, 8, 1.0, candidate_miss=True),  # Reference is present.
        CandidateScore("trip", 3, 12, 0.9),
        CandidateScore("trip", 3, 9, 0.1),
    ]
    reference = {0: 8, 1: 8, 2: 9, 3: 9, 4: 9}
    # Points 0 and 2 miss. Point 4 is unusable and point 5 has no reference.
    assert final_candidate_miss_rate(points, candidates, reference) == 0.5
