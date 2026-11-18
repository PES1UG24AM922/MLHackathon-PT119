import numpy as np
import pytest

from pathsnap.decoder.graph import RoadGraph


def test_directed_shortest_path_and_parallel_arcs():
    graph = RoadGraph(
        np.array([[0, 0], [10, 0], [20, 0], [0, 10]], dtype=float),
        np.array([[0, 1], [0, 1], [1, 2], [3, 3]], dtype=int),
    )
    distance, nodes, arcs = graph.shortest_node_path(0, 2)  # type: ignore[misc]
    assert distance == 20.0
    assert nodes == [0, 1, 2]
    assert arcs == [0, 2]  # deterministic lowest-ID parallel arc
    assert graph.shortest_node_path(2, 0) is None
    assert graph.arc(3).length_m == 0.0


def test_graph_rejects_invalid_endpoint():
    with pytest.raises(ValueError, match="out-of-range"):
        RoadGraph(np.array([[0, 0]], dtype=float), np.array([[0, 1]], dtype=int))
