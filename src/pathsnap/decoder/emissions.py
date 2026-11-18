"""Candidate-score validation and stable HMM emission preparation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .config import DecoderConfig
from .schemas import CandidateScore, DecoderWarning


@dataclass(frozen=True)
class EmissionState:
    arc_id: int
    probability: float
    log_emission: float
    projection_ratio: float | None


def prepare_emissions(
    candidates: Iterable[CandidateScore], config: DecoderConfig
) -> tuple[dict[int, list[EmissionState]], list[DecoderWarning]]:
    """Sanitize, deduplicate and normalize probabilities per GPS observation.

    Candidate duplicates occur naturally when spatial indexes combine radius and
    k-nearest-neighbour queries.  Keeping the highest score is deterministic
    and avoids duplicate HMM states.
    """
    grouped: dict[int, dict[int, CandidateScore]] = defaultdict(dict)
    warnings: list[DecoderWarning] = []
    trip_ids = set()
    for candidate in candidates:
        trip_ids.add(candidate.trip_id)
        probability = candidate.probability
        if not math.isfinite(probability) or probability <= 0:
            warnings.append(
                DecoderWarning(
                    "INVALID_PROBABILITY_CLIPPED",
                    candidate.gps_point_idx,
                    f"Arc {candidate.candidate_arc_id} probability was {probability!r}; clipped.",
                )
            )
            probability = config.probability_floor
        normalized_candidate = CandidateScore(
            trip_id=candidate.trip_id,
            gps_point_idx=candidate.gps_point_idx,
            candidate_arc_id=candidate.candidate_arc_id,
            probability=probability,
            candidate_miss=candidate.candidate_miss,
            projection_ratio=candidate.projection_ratio,
        )
        existing = grouped[candidate.gps_point_idx].get(candidate.candidate_arc_id)
        if existing is not None:
            warnings.append(
                DecoderWarning(
                    "DUPLICATE_CANDIDATE_DEDUPLICATED",
                    candidate.gps_point_idx,
                    f"Arc {candidate.candidate_arc_id} appeared more than once; highest score kept.",
                )
            )
            normalized_candidate = max(
                (existing, normalized_candidate), key=lambda value: value.probability
            )
        grouped[candidate.gps_point_idx][candidate.candidate_arc_id] = normalized_candidate

    if len(trip_ids) > 1:
        raise ValueError("one decode call must contain candidates for exactly one trip")

    emissions: dict[int, list[EmissionState]] = {}
    for gps_idx, scores in grouped.items():
        total = sum(candidate.probability for candidate in scores.values())
        if not math.isfinite(total) or total <= 0:
            warnings.append(
                DecoderWarning(
                    "INVALID_GROUP_NORMALIZED_UNIFORM",
                    gps_idx,
                    "Candidate score group had no positive finite total.",
                )
            )
            total = float(len(scores))
            scores = {
                arc_id: CandidateScore(
                    trip_id=candidate.trip_id,
                    gps_point_idx=candidate.gps_point_idx,
                    candidate_arc_id=arc_id,
                    probability=1.0,
                    candidate_miss=candidate.candidate_miss,
                    projection_ratio=candidate.projection_ratio,
                )
                for arc_id, candidate in scores.items()
            }
        states = []
        for arc_id in sorted(scores):
            candidate = scores[arc_id]
            normalized = max(candidate.probability / total, config.probability_floor)
            states.append(
                EmissionState(arc_id, normalized, math.log(normalized), candidate.projection_ratio)
            )
        emissions[gps_idx] = states
    return emissions, warnings


def classifier_only_predictions(emissions: dict[int, list[EmissionState]]) -> dict[int, int]:
    """Deterministic argmax baseline for the compulsory HMM comparison."""
    result: dict[int, int] = {}
    for gps_idx, states in emissions.items():
        if states:
            result[gps_idx] = min(states, key=lambda state: (-state.probability, state.arc_id)).arc_id
    return result
