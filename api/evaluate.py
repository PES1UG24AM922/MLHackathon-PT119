"""Reference route scoring, called only after label-free prediction."""
from itertools import chain

import numpy as np

from src.features.candidate_gen import assign_ground_truth_arcs, projected_coordinates


def ordered_route_metrics(reference, arc_segments):
    predicted = list(chain.from_iterable(arc_segments))
    # Bit-parallel longest common subsequence, preserving order and multiplicity.
    masks = {}
    for index, arc in enumerate(reference):
        masks[arc] = masks.get(arc, 0) | (1 << index)
    row = 0
    for arc in predicted:
        matches = row | masks.get(arc, 0)
        row = matches & ~(matches - ((row << 1) | 1))
    common = row.bit_count()
    return {
        "ordered_route_recall": common / len(reference),
        "ordered_route_precision": common / len(predicted) if predicted else 0.0,
        "ordered_route_f1": 2 * common / (len(reference) + len(predicted)),
    }


def evaluate_prediction(store, trip, point_arcs, arc_segments):
    reference = store.read_reference(trip.trip_id, len(trip.arcs))
    labels = assign_ground_truth_arcs(
        *projected_coordinates(trip),
        trip.arcs["from_node"].to_numpy(dtype=int),
        trip.arcs["to_node"].to_numpy(dtype=int),
        np.asarray(reference, dtype=int),
        trip.track["timestamp"].to_numpy(dtype=float),
    )
    predicted = np.array([-1 if arc is None else arc for arc in point_arcs])
    return {
        **ordered_route_metrics(reference, arc_segments),
        "point_top1_accuracy": float(np.mean(predicted == labels)),
    }
