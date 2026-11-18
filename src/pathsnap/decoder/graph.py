"""Directed road graph with deterministic, cached shortest-path queries."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np

from .geometry import bearing_radians, euclidean_m


@dataclass(frozen=True)
class Arc:
    arc_id: int
    from_node_id: int
    to_node_id: int
    length_m: float
    bearing_rad: float | None


class RoadGraph:
    """A directed graph indexed by original arc ID, including parallel arcs.

    Nodes must be projected metre coordinates in `(x, y)` order.  Any WGS84
    conversion belongs in Person 1's preprocessing artifact, never here.
    """

    def __init__(
        self,
        nodes_xy_m: np.ndarray,
        arcs: np.ndarray,
        nodes_lonlat: np.ndarray | None = None,
    ) -> None:
        nodes = np.asarray(nodes_xy_m, dtype=float)
        edge_rows = np.asarray(arcs, dtype=int)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes_xy_m must have shape (n_nodes, 2)")
        if edge_rows.ndim != 2 or edge_rows.shape[1] != 2:
            raise ValueError("arcs must have shape (n_arcs, 2)")
        if not np.isfinite(nodes).all():
            raise ValueError("nodes_xy_m contains non-finite coordinates")
        if edge_rows.size and ((edge_rows < 0).any() or (edge_rows >= len(nodes)).any()):
            raise ValueError("arcs contains an out-of-range node ID")

        if nodes_lonlat is not None:
            lonlat = np.asarray(nodes_lonlat, dtype=float)
            if lonlat.shape != nodes.shape or not np.isfinite(lonlat).all():
                raise ValueError("nodes_lonlat must be finite with shape (n_nodes, 2)")
            self.nodes_lonlat: np.ndarray | None = lonlat
        else:
            self.nodes_lonlat = None
        self.nodes_xy_m = nodes
        self._arcs: dict[int, Arc] = {}
        self._adjacency: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
        self._node_path_cache: dict[tuple[int, int, float], tuple[float, list[int], list[int]] | None] = {}

        for arc_id, (from_node, to_node) in enumerate(edge_rows):
            x1, y1 = nodes[from_node]
            x2, y2 = nodes[to_node]
            length = euclidean_m(float(x1), float(y1), float(x2), float(y2))
            arc = Arc(
                arc_id=arc_id,
                from_node_id=int(from_node),
                to_node_id=int(to_node),
                length_m=length,
                bearing_rad=bearing_radians(float(x1), float(y1), float(x2), float(y2)),
            )
            self._arcs[arc_id] = arc
            self._adjacency[arc.from_node_id].append((arc.to_node_id, arc_id, arc.length_m))

        for outgoing in self._adjacency.values():
            outgoing.sort(key=lambda edge: edge[1])

    @property
    def n_nodes(self) -> int:
        return len(self.nodes_xy_m)

    @property
    def n_arcs(self) -> int:
        return len(self._arcs)

    def has_arc(self, arc_id: int) -> bool:
        return arc_id in self._arcs

    def arc(self, arc_id: int) -> Arc:
        try:
            return self._arcs[arc_id]
        except KeyError as exc:
            raise ValueError(f"unknown arc ID: {arc_id}") from exc

    def arc_geometry_xy(self, arc_id: int) -> list[tuple[float, float]]:
        arc = self.arc(arc_id)
        return [
            tuple(map(float, self.nodes_xy_m[arc.from_node_id])),
            tuple(map(float, self.nodes_xy_m[arc.to_node_id])),
        ]

    def arc_geometry_latlon(self, arc_id: int) -> list[tuple[float, float]]:
        """Return `(latitude, longitude)` for Folium when WGS84 nodes exist."""
        if self.nodes_lonlat is None:
            raise ValueError("graph has no WGS84 node geometry for map rendering")
        arc = self.arc(arc_id)
        # Stored as longitude, latitude to match raw PathSnap files.
        return [
            (float(self.nodes_lonlat[arc.from_node_id, 1]), float(self.nodes_lonlat[arc.from_node_id, 0])),
            (float(self.nodes_lonlat[arc.to_node_id, 1]), float(self.nodes_lonlat[arc.to_node_id, 0])),
        ]

    def shortest_node_path(
        self,
        source_node_id: int,
        target_node_id: int,
        max_distance_m: float = math.inf,
    ) -> tuple[float, list[int], list[int]] | None:
        """Dijkstra result: distance, nodes, arcs; None when unreachable/capped."""
        if not (0 <= source_node_id < self.n_nodes and 0 <= target_node_id < self.n_nodes):
            raise ValueError("source and target node IDs must exist")
        cache_key = (source_node_id, target_node_id, float(max_distance_m))
        if cache_key in self._node_path_cache:
            return self._node_path_cache[cache_key]
        if source_node_id == target_node_id:
            result = (0.0, [source_node_id], [])
            self._node_path_cache[cache_key] = result
            return result

        queue: list[tuple[float, int]] = [(0.0, source_node_id)]
        distances = {source_node_id: 0.0}
        previous: dict[int, tuple[int, int]] = {}
        found = False
        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance != distances.get(node):
                continue
            if current_distance > max_distance_m:
                break
            if node == target_node_id:
                found = True
                break
            for neighbour, arc_id, length_m in self._adjacency.get(node, []):
                candidate_distance = current_distance + length_m
                if candidate_distance > max_distance_m:
                    continue
                # Strictly lower distance preserves first (lowest-ID) arc on ties.
                if candidate_distance < distances.get(neighbour, math.inf):
                    distances[neighbour] = candidate_distance
                    previous[neighbour] = (node, arc_id)
                    heapq.heappush(queue, (candidate_distance, neighbour))
        if not found:
            self._node_path_cache[cache_key] = None
            return None

        nodes = [target_node_id]
        path_arcs: list[int] = []
        node = target_node_id
        while node != source_node_id:
            prior_node, arc_id = previous[node]
            path_arcs.append(arc_id)
            nodes.append(prior_node)
            node = prior_node
        nodes.reverse()
        path_arcs.reverse()
        result = (distances[target_node_id], nodes, path_arcs)
        self._node_path_cache[cache_key] = result
        return result

    def arc_transition_path(
        self,
        from_arc_id: int,
        to_arc_id: int,
        max_distance_m: float = math.inf,
    ) -> tuple[float, list[int]] | None:
        """Route from the exit of first arc to entry of second, including both arcs.

        This is map-display topology, not an exact vehicle position estimate.
        """
        from_arc = self.arc(from_arc_id)
        to_arc = self.arc(to_arc_id)
        if from_arc_id == to_arc_id:
            return (0.0, [from_arc_id])
        middle = self.shortest_node_path(
            from_arc.to_node_id, to_arc.from_node_id, max_distance_m
        )
        if middle is None:
            return None
        distance, _, middle_arcs = middle
        return (distance, [from_arc_id, *middle_arcs, to_arc_id])

    def transition_network_distance(
        self,
        from_arc_id: int,
        to_arc_id: int,
        max_distance_m: float,
    ) -> float | None:
        """Conservative centre-to-centre approximation between two arc states."""
        if from_arc_id == to_arc_id:
            return 0.0
        from_arc = self.arc(from_arc_id)
        to_arc = self.arc(to_arc_id)
        middle = self.shortest_node_path(
            from_arc.to_node_id, to_arc.from_node_id, max_distance_m
        )
        if middle is None:
            return None
        return from_arc.length_m / 2.0 + middle[0] + to_arc.length_m / 2.0

    def clear_cache(self) -> None:
        self._node_path_cache.clear()
