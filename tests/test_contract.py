import numpy as np
import pytest

from pathsnap.decoder import CandidateScore, GPSPoint, HMMDecoder, RoadGraph
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.emissions import prepare_emissions


def _graph():
    return RoadGraph(np.array([[0, 0], [10, 0]], dtype=float), np.array([[0, 1]], dtype=int))


def test_unordered_or_duplicate_gps_indices_are_rejected():
    decoder = HMMDecoder(_graph())
    candidates = [CandidateScore("trip", 0, 0, 1.0)]
    with pytest.raises(ValueError, match="sorted"):
        decoder.decode("trip", [GPSPoint(1, 1, 1, 0), GPSPoint(0, 0, 0, 0)], candidates)
    with pytest.raises(ValueError, match="unique"):
        decoder.decode("trip", [GPSPoint(0, 0, 0, 0), GPSPoint(0, 1, 1, 0)], candidates)


def test_candidate_must_reference_an_existing_gps_point():
    with pytest.raises(ValueError, match="missing GPS"):
        HMMDecoder(_graph()).decode(
            "trip", [GPSPoint(0, 0, 0, 0)], [CandidateScore("trip", 9, 0, 1.0)]
        )


def test_mixed_trip_emissions_are_rejected():
    with pytest.raises(ValueError, match="exactly one trip"):
        prepare_emissions(
            [CandidateScore("a", 0, 0, 1.0), CandidateScore("b", 0, 0, 1.0)], DecoderConfig()
        )


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="probability_floor"):
        DecoderConfig(probability_floor=0.0)
    with pytest.raises(ValueError, match="speed"):
        DecoderConfig(max_plausible_speed_mps=0.0)
