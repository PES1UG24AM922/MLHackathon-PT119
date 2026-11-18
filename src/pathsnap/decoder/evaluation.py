"""Offline-only decoder evaluation. Keep labels out of inference code paths."""

from __future__ import annotations

import math
from typing import Iterable

from .graph import RoadGraph
from .schemas import CandidateScore, DecodeResult, GPSPoint


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
    segment_ids: dict[int, int] | None = None,
) -> dict[str, float | int]:
    """Evaluate every eligible aligned observation, including abstentions.

    An unmatched prediction is counted as incorrect and included in coverage;
    compact display routes are never used as an evaluation denominator.
    """
    point_by_idx = {point.gps_point_idx: point for point in gps_points}
    eligible = [
        (gps_idx, reference)
        for gps_idx, reference in reference_labels.items()
        if gps_idx in point_by_idx and point_by_idx[gps_idx].usable
    ]
    if not eligible:
        return {
            "n_evaluated": 0,
            "n_predicted": 0,
            "n_unmatched": 0,
            "prediction_coverage": float("nan"),
            "top1_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "route_continuity": float("nan"),
            "continuity_breaks": 0,
            "candidate_miss_rate": candidate_miss_rate if candidate_miss_rate is not None else float("nan"),
        }
    y_true = [reference for _, reference in eligible]
    unmatched_sentinel = -1
    y_pred = [
        int(decoded_arc_ids.get(gps_idx))
        if decoded_arc_ids.get(gps_idx) is not None
        else unmatched_sentinel
        for gps_idx, _ in eligible
    ]
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    n_predicted = sum(prediction != unmatched_sentinel for prediction in y_pred)
    ordered = sorted((idx, int(arc)) for idx, arc in decoded_arc_ids.items() if arc is not None and idx in point_by_idx)
    continuity_checks = 0
    connected = 0
    continuity_breaks = 0
    for (previous_idx, previous_arc), (current_idx, current_arc) in zip(ordered, ordered[1:]):
        previous_point = point_by_idx[previous_idx]
        current_point = point_by_idx[current_idx]
        dt = current_point.timestamp_s - previous_point.timestamp_s
        is_segment_break = (
            not math.isfinite(dt)
            or dt <= 0
            or (segment_ids is not None and segment_ids.get(previous_idx) != segment_ids.get(current_idx))
            or current_idx != previous_idx + 1
        )
        if is_segment_break:
            continuity_breaks += 1
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
        "n_evaluated": len(eligible),
        "n_predicted": n_predicted,
        "n_unmatched": len(eligible) - n_predicted,
        "prediction_coverage": n_predicted / len(eligible),
        "top1_accuracy": accuracy,
        "macro_f1": macro_f1(y_true, y_pred),
        "route_continuity": connected / continuity_checks if continuity_checks else float("nan"),
        "continuity_breaks": continuity_breaks,
        "candidate_miss_rate": candidate_miss_rate if candidate_miss_rate is not None else float("nan"),
    }


def candidate_miss_rate(candidate_miss_flags: Iterable[bool]) -> float:
    flags = list(candidate_miss_flags)
    return sum(flags) / len(flags) if flags else float("nan")


def final_candidate_miss_rate(
    gps_points: Iterable[GPSPoint],
    candidate_scores: Iterable[CandidateScore],
    reference_labels: dict[int, int],
) -> float:
    """Measure whether each eligible reference arc is in its final candidate set."""
    by_point: dict[int, set[int]] = {}
    for score in candidate_scores:
        by_point.setdefault(score.gps_point_idx, set()).add(score.candidate_arc_id)
    return candidate_miss_rate(
        reference_labels[point.gps_point_idx] not in by_point.get(point.gps_point_idx, ())
        for point in gps_points
        if point.usable and point.gps_point_idx in reference_labels
    )


def result_prediction_map(result: DecodeResult) -> dict[int, int | None]:
    return {point.gps_point_idx: point.decoded_arc_id for point in result.decoded_points}


def result_segment_map(result: DecodeResult) -> dict[int, int]:
    return {point.gps_point_idx: point.segment_id for point in result.decoded_points}


def prediction_segment_map(
    gps_points: Iterable[GPSPoint],
    predictions: dict[int, int | None],
    gap_reset_seconds: float,
) -> dict[int, int]:
    """Apply decoder-equivalent gap/missing rules to a classifier baseline."""
    segments: dict[int, int] = {}
    segment_id = 0
    previous: GPSPoint | None = None
    previous_has_prediction = False
    for point in sorted(gps_points, key=lambda value: value.gps_point_idx):
        has_prediction = point.usable and predictions.get(point.gps_point_idx) is not None
        if has_prediction:
            dt = point.timestamp_s - previous.timestamp_s if previous is not None else None
            if (
                previous is None
                or not previous_has_prediction
                or dt is None
                or not math.isfinite(dt)
                or dt <= 0
                or dt > gap_reset_seconds
            ):
                if segments:
                    segment_id += 1
            segments[point.gps_point_idx] = segment_id
        previous = point
        previous_has_prediction = has_prediction
    return segments
