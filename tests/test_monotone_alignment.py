import numpy as np

from src.features.candidate_gen import assign_ground_truth_arcs


def test_out_and_back_uses_both_directed_occurrences():
    x = np.array([2., 7., 9., 8., 5., 1.])
    actual = assign_ground_truth_arcs(
        x, np.zeros_like(x), np.array([0., 10.]), np.zeros(2),
        np.array([0, 1]), np.array([1, 0]), np.array([0, 1]), np.arange(len(x))
    )
    assert actual[0] == 0 and actual[-1] == 1
    assert np.all(np.diff(actual) >= 0)
