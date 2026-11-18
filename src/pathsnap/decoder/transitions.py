"""Topology-aware HMM transition scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import DecoderConfig
from .geometry import euclidean_m
from .graph import RoadGraph
from .schemas import GPSPoint


@dataclass(frozen=True)
class TransitionScore:
    log_probability: float
    network_distance_m: float | None
    disconnected: bool
    excessive_speed: bool


class TransitionModel:
    """Scores arc-to-arc movement without using reference labels."""

    def __init__(self, graph: RoadGraph, config: DecoderConfig) -> None:
        self.graph = graph
        self.config = config

    def score(
        self,
        previous_point: GPSPoint,
        current_point: GPSPoint,
        from_arc_id: int,
        to_arc_id: int,
    ) -> TransitionScore:
        dt = current_point.timestamp_s - previous_point.timestamp_s
        if not math.isfinite(dt) or dt <= 0:
            # Decoder starts a new segment for this case. This value is safe if
            # callers inspect transitions independently.
            return TransitionScore(
                self.config.disconnected_transition_log_penalty,
                None,
                True,
                False,
            )
        observed_distance = euclidean_m(
            previous_point.x_m, previous_point.y_m, current_point.x_m, current_point.y_m
        )
        # Network search only needs consider feasible paths plus a tolerance.
        search_cap = min(
            self.config.max_network_search_m,
            max(observed_distance * 4.0 + 200.0, self.config.max_plausible_speed_mps * dt * 1.5),
        )
        network_distance = self.graph.transition_network_distance(
            from_arc_id, to_arc_id, search_cap
        )
        if network_distance is None:
            return TransitionScore(
                self.config.disconnected_transition_log_penalty,
                None,
                True,
                False,
            )

        if from_arc_id == to_arc_id:
            # Exact intra-arc position is unknown. Holding the same directed arc
            # is normally plausible, so avoid inventing a half-arc movement.
            network_distance = observed_distance
        residual = abs(network_distance - observed_distance)
        log_score = -residual / self.config.transition_distance_scale_m
        implied_speed = network_distance / dt
        excessive_speed = implied_speed > self.config.max_plausible_speed_mps
        if excessive_speed:
            speed_ratio = implied_speed / self.config.max_plausible_speed_mps
            log_score -= self.config.speed_penalty * (speed_ratio - 1.0)
        return TransitionScore(log_score, network_distance, False, excessive_speed)
