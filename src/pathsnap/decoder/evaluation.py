"""Offline-only decoder evaluation. Keep labels out of inference code paths."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

from .graph import RoadGraph
from .schemas import DecodeResult, GPSPoint


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return float("nan")
    labels = sorted(set(y_true) | set(y_pred))
    per_class = []
    for label in labels:
        true_positive = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        false_positive = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        false_negative = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(per_class) / len(per_class)


def evaluate_predictions(
    decoded_arc_ids: dict[int, int | None],
    reference_labels: dict[int, int],
    graph: RoadGraph,
    gps_points: Iterable[GPSPoint],
    max_network_distance_m: float = 20_000.0,
    candidate_miss_rate: float | None = None,
) -> dict[str, float | int]:
    """Evaluate only aligned point-level states, never compact display routes."""
    comparable = [
        (gps_idx, reference)
        for gps_idx, reference in reference_labels.items()
        if gps_idx in decoded_arc_ids and decoded_arc_ids[gps_idx] is not None
    ]
    if not comparable:
        return {
            "n_evaluated": 0,
            "top1_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "route_continuity": float("nan"),
            "candidate_miss_rate": candidate_miss_rate if candidate_miss_rate is not None else float("nan"),
        }
    y_true = [reference for _, reference in comparable]
    y_pred = [int(decoded_arc_ids[gps_idx]) for gps_idx, _ in comparable]
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

    point_by_idx = {point.gps_point_idx: point for point in gps_points}
    ordered = sorted(
        (idx, arc) for idx, arc in decoded_arc_ids.items() if arc is not None and idx in point_by_idx
    )
    continuity_checks = 0
    connected = 0
    for (previous_idx, previous_arc), (current_idx, current_arc) in zip(ordered, ordered[1:]):
        previous_point = point_by_idx[previous_idx]
        current_point = point_by_idx[current_idx]
        dt = current_point.timestamp_s - previous_point.timestamp_s
        if not math.isfinite(dt) or dt <= 0:
            continue
        if previous_arc == current_arc:
            connected += 1
            continuity_checks += 1
            continue
        path = graph.transition_network_distance(
            previous_arc, current_arc, max_network_distance_m
        )
        continuity_checks += 1
        if path is not None:
            connected += 1
    return {
        "n_evaluated": len(comparable),
        "top1_accuracy": accuracy,
        "macro_f1": macro_f1(y_true, y_pred),
        "route_continuity": connected / continuity_checks if continuity_checks else float("nan"),
        "candidate_miss_rate": candidate_miss_rate if candidate_miss_rate is not None else float("nan"),
    }


def candidate_miss_rate(candidate_miss_flags: Iterable[bool]) -> float:
    flags = list(candidate_miss_flags)
    return sum(flags) / len(flags) if flags else float("nan")


def result_prediction_map(result: DecodeResult) -> dict[int, int | None]:
    return {point.gps_point_idx: point.decoded_arc_id for point in result.decoded_points}
