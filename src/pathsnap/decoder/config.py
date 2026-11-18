"""Configuration for numerically stable, directed HMM decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DecoderConfig:
    """Versioned decoder parameters.

    The configuration is deliberately data-only so the same artifact can be
    loaded by offline evaluation, tuning, and the FastAPI inference process.
    """

    schema_version: str = "1.0"
    decoder_version: str = "1.0.0"
    probability_floor: float = 1e-12
    emission_weight: float = 1.0
    initial_emission_weight: float = 1.0
    transition_distance_scale_m: float = 35.0
    max_plausible_speed_mps: float = 55.0
    speed_penalty: float = 3.0
    disconnected_transition_log_penalty: float = -35.0
    gap_reset_seconds: float = 30.0
    max_network_search_m: float = 20_000.0
    tie_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.probability_floor <= 0 or self.probability_floor >= 1:
            raise ValueError("probability_floor must lie strictly between 0 and 1")
        if self.transition_distance_scale_m <= 0:
            raise ValueError("transition_distance_scale_m must be positive")
        if self.max_plausible_speed_mps <= 0:
            raise ValueError("max_plausible_speed_mps must be positive")
        if self.gap_reset_seconds < 0:
            raise ValueError("gap_reset_seconds cannot be negative")

    @classmethod
    def from_file(cls, path: str | Path) -> "DecoderConfig":
        """Read JSON. JSON is also valid YAML, so `hmm.yaml` needs no parser."""
        with Path(path).open(encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
