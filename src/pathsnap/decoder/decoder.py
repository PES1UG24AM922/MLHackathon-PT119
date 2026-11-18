"""Segmented, directed HMM/Viterbi route decoder."""

from __future__ import annotations

import math
import time
from typing import Iterable

from .config import DecoderConfig
from .emissions import EmissionState, prepare_emissions
from .graph import RoadGraph
from .schemas import (
    CandidateScore,
    DecodeResult,
    DecodedPoint,
    DecoderWarning,
    GPSPoint,
)
from .transitions import TransitionModel


class HMMDecoder:
    """Decode calibrated candidate scores while preserving road direction.

    The class has no label/reference-route argument by design.  Ground truth is
    accepted only by `evaluation.py`, which should run offline/server-side.
    """

    def __init__(self, graph: RoadGraph, config: DecoderConfig | None = None) -> None:
        self.graph = graph
        self.config = config or DecoderConfig()
        self.transitions = TransitionModel(graph, self.config)

    def decode(
        self,
        trip_id: str,
        gps_points: Iterable[GPSPoint],
        candidate_scores: Iterable[CandidateScore],
    ) -> DecodeResult:
        started = time.perf_counter()
        points = list(gps_points)
        self._validate_points(points)
        candidates = list(candidate_scores)
        if any(candidate.trip_id != trip_id for candidate in candidates):
            raise ValueError("candidate trip_id does not match requested trip_id")
        emissions, warnings = prepare_emissions(candidates, self.config)
        point_by_idx = {point.gps_point_idx: point for point in points}
        for gps_idx, states in emissions.items():
            if gps_idx not in point_by_idx:
                raise ValueError(f"candidate references missing GPS point index {gps_idx}")
            for state in states:
                if not self.graph.has_arc(state.arc_id):
                    raise ValueError(f"candidate references unknown arc ID {state.arc_id}")

        decoded: dict[int, tuple[int | None, float | None, int]] = {}
        segment: list[GPSPoint] = []
        segment_id = 0
        previous_point: GPSPoint | None = None
        for point in points:
            reset_reason = self._reset_reason(previous_point, point, emissions)
            if reset_reason and segment:
                self._decode_segment(segment, emissions, decoded, segment_id, warnings)
                segment = []
                segment_id += 1
            if reset_reason:
                warnings.append(DecoderWarning(reset_reason, point.gps_point_idx, "Decoder segment reset."))
            if not point.usable:
                decoded[point.gps_point_idx] = (None, None, segment_id)
                warnings.append(
                    DecoderWarning("UNUSABLE_GPS_POINT", point.gps_point_idx, "Point marked unusable by preprocessing.")
                )
                previous_point = point
                continue
            if not emissions.get(point.gps_point_idx):
                decoded[point.gps_point_idx] = (None, None, segment_id)
                warnings.append(
                    DecoderWarning("EMPTY_CANDIDATE_SET", point.gps_point_idx, "No candidate arcs; point left unmatched.")
                )
                previous_point = point
                continue
            segment.append(point)
            previous_point = point
        if segment:
            self._decode_segment(segment, emissions, decoded, segment_id, warnings)

        decoded_points = [
            DecodedPoint(point.gps_point_idx, *decoded.get(point.gps_point_idx, (None, None, segment_id)))
            for point in points
        ]
        compact = self._compact_route(decoded_points)
        expanded = self._expand_route(compact, warnings)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DecodeResult(
            trip_id=trip_id,
            decoded_points=decoded_points,
            compact_route_arc_ids=compact,
            expanded_route_arc_ids=expanded,
            warnings=warnings,
            inference_time_ms=elapsed_ms,
            decoder_version=self.config.decoder_version,
        )

    def _validate_points(self, points: list[GPSPoint]) -> None:
        indices = [point.gps_point_idx for point in points]
        if len(indices) != len(set(indices)):
            raise ValueError("GPS point indices must be unique")
        if indices != sorted(indices):
            raise ValueError("GPS points must be sorted by gps_point_idx")

    def _reset_reason(
        self,
        previous_point: GPSPoint | None,
        point: GPSPoint,
        emissions: dict[int, list[EmissionState]],
    ) -> str | None:
        if previous_point is None:
            return "INITIAL_SEGMENT"
        if not previous_point.usable:
            return "PREVIOUS_GPS_UNUSABLE"
        if not point.usable:
            return "GPS_UNUSABLE"
        if not emissions.get(previous_point.gps_point_idx):
            return "PREVIOUS_EMPTY_CANDIDATE_SET"
        if not emissions.get(point.gps_point_idx):
            return "EMPTY_CANDIDATE_SET"
        dt = point.timestamp_s - previous_point.timestamp_s
        if not math.isfinite(dt) or dt <= 0:
            return "NON_INCREASING_TIMESTAMP"
        if dt > self.config.gap_reset_seconds:
            return "LARGE_GPS_GAP"
        return None

    def _decode_segment(
        self,
        points: list[GPSPoint],
        emissions: dict[int, list[EmissionState]],
        decoded: dict[int, tuple[int | None, float | None, int]],
        segment_id: int,
        warnings: list[DecoderWarning],
    ) -> None:
        if not points:
            return
        scores: list[dict[int, float]] = []
        backpointers: list[dict[int, int | None]] = []
        first_states = emissions[points[0].gps_point_idx]
        first_scores = {
            state.arc_id: self.config.initial_emission_weight * state.log_emission
            for state in first_states
        }
        scores.append(first_scores)
        backpointers.append({state.arc_id: None for state in first_states})

        for time_idx in range(1, len(points)):
            prior = points[time_idx - 1]
            current = points[time_idx]
            current_scores: dict[int, float] = {}
            current_backpointers: dict[int, int | None] = {}
            for current_state in emissions[current.gps_point_idx]:
                best_score = -math.inf
                best_prior_arc: int | None = None
                disconnected_count = 0
                for prior_arc, prior_score in scores[-1].items():
                    transition = self.transitions.score(
                        prior, current, prior_arc, current_state.arc_id
                    )
                    if transition.disconnected:
                        disconnected_count += 1
                    candidate_score = (
                        prior_score
                        + transition.log_probability
                        + self.config.emission_weight * current_state.log_emission
                    )
                    if self._better(candidate_score, prior_arc, best_score, best_prior_arc):
                        best_score = candidate_score
                        best_prior_arc = prior_arc
                if best_prior_arc is None:
                    # Defensive fallback; should only happen with malformed state input.
                    best_score = self.config.emission_weight * current_state.log_emission
                    warnings.append(
                        DecoderWarning("VITERBI_REINITIALIZED", current.gps_point_idx, "No predecessor state available.")
                    )
                elif disconnected_count == len(scores[-1]):
                    warnings.append(
                        DecoderWarning(
                            "ALL_PREDECESSORS_DISCONNECTED",
                            current.gps_point_idx,
                            "Used the least-penalized disconnected transition.",
                        )
                    )
                current_scores[current_state.arc_id] = best_score
                current_backpointers[current_state.arc_id] = best_prior_arc
            scores.append(current_scores)
            backpointers.append(current_backpointers)

        last_arc = min(scores[-1], key=lambda arc_id: (-scores[-1][arc_id], arc_id))
        decoded[points[-1].gps_point_idx] = (last_arc, scores[-1][last_arc], segment_id)
        for time_idx in range(len(points) - 1, 0, -1):
            predecessor = backpointers[time_idx][last_arc]
            if predecessor is None:
                warnings.append(
                    DecoderWarning("VITERBI_BACKPOINTER_MISSING", points[time_idx].gps_point_idx, "Sequence reinitialized.")
                )
                predecessor = min(scores[time_idx - 1], key=lambda arc_id: (-scores[time_idx - 1][arc_id], arc_id))
            last_arc = predecessor
            decoded[points[time_idx - 1].gps_point_idx] = (
                last_arc,
                scores[time_idx - 1][last_arc],
                segment_id,
            )

    def _better(
        self,
        candidate_score: float,
        candidate_arc: int,
        best_score: float,
        best_arc: int | None,
    ) -> bool:
        if candidate_score > best_score + self.config.tie_tolerance:
            return True
        return (
            best_arc is not None
            and abs(candidate_score - best_score) <= self.config.tie_tolerance
            and candidate_arc < best_arc
        )

    def _compact_route(self, decoded_points: list[DecodedPoint]) -> list[int]:
        route: list[int] = []
        for point in decoded_points:
            if point.decoded_arc_id is not None and (not route or route[-1] != point.decoded_arc_id):
                route.append(point.decoded_arc_id)
        return route

    def _expand_route(self, compact: list[int], warnings: list[DecoderWarning]) -> list[int]:
        if not compact:
            return []
        expanded = [compact[0]]
        for previous_arc, current_arc in zip(compact, compact[1:]):
            path = self.graph.arc_transition_path(
                previous_arc, current_arc, self.config.max_network_search_m
            )
            if path is None:
                warnings.append(
                    DecoderWarning(
                        "DISPLAY_ROUTE_DISCONNECTED",
                        None,
                        f"Cannot expand display route from arc {previous_arc} to {current_arc}.",
                    )
                )
                expanded.append(current_arc)
                continue
            expanded.extend(path[1][1:])
        return expanded
