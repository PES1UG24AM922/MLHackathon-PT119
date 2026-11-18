"""Reference route scoring, called only after label-free prediction."""
from itertools import chain


def evaluate_prediction(store, trip_id, n_arcs, arc_segments):
    reference = store.read_reference(trip_id, n_arcs)
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
