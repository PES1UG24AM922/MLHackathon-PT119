"""Strict, label-free inference contracts shared by Persons 1--4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GPSPoint:
    """One ordered GPS observation in projected metres and WGS84 degrees."""

    gps_point_idx: int
    timestamp_s: float
    x_m: float
    y_m: float
    longitude: float | None = None
    latitude: float | None = None
    usable: bool = True


@dataclass(frozen=True)
class CandidateScore:
    """Calibrated model score for one directed arc candidate.

    This intentionally contains no reference/ground-truth label.  The same
    structure is safe to pass into production inference and the evaluator UI.
    """

    trip_id: str
    gps_point_idx: int
    candidate_arc_id: int
    probability: float
    candidate_miss: bool = False


@dataclass(frozen=True)
class DecoderWarning:
    code: str
    gps_point_idx: int | None
    detail: str


@dataclass(frozen=True)
class DecodedPoint:
    gps_point_idx: int
    decoded_arc_id: int | None
    log_score: float | None
    segment_id: int


@dataclass
class DecodeResult:
    """Inference result. `decoded_points` retains one row per GPS input."""

    trip_id: str
    decoded_points: list[DecodedPoint]
    compact_route_arc_ids: list[int]
    expanded_route_arc_ids: list[int]
    warnings: list[DecoderWarning] = field(default_factory=list)
    inference_time_ms: float = 0.0
    decoder_version: str = ""

    @property
    def decoded_arc_ids(self) -> list[Optional[int]]:
        return [point.decoded_arc_id for point in self.decoded_points]
