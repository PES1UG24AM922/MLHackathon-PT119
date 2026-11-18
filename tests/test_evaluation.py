import numpy as np

from pathsnap.decoder.evaluation import evaluate_predictions, macro_f1
from pathsnap.decoder.graph import RoadGraph
from pathsnap.decoder.schemas import GPSPoint


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
