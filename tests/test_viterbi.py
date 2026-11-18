import numpy as np

from pathsnap.decoder import CandidateScore, DecoderConfig, GPSPoint, HMMDecoder, RoadGraph


def _graph():
    # 0 -> 1 -> 2 is the valid road; 3 -> 4 is disconnected.
    return RoadGraph(
        np.array([[0, 0], [10, 0], [20, 0], [0, 20], [10, 20]], dtype=float),
        np.array([[0, 1], [1, 2], [3, 4]], dtype=int),
    )


def test_viterbi_prefers_connected_route_over_high_disconnected_emission():
    decoder = HMMDecoder(_graph(), DecoderConfig(transition_distance_scale_m=5.0))
    points = [GPSPoint(0, 0, 1, 0), GPSPoint(1, 1, 11, 0)]
    candidates = [
        CandidateScore("trip", 0, 0, 0.9),
        CandidateScore("trip", 0, 2, 0.1),
        CandidateScore("trip", 1, 1, 0.35),
        CandidateScore("trip", 1, 2, 0.65),
    ]
    result = decoder.decode("trip", points, candidates)
    assert result.decoded_arc_ids == [0, 1]
    assert result.compact_route_arc_ids == [0, 1]


def test_tie_break_is_deterministic():
    decoder = HMMDecoder(_graph())
    point = [GPSPoint(0, 0, 1, 0)]
    candidates = [CandidateScore("trip", 0, 2, 0.5), CandidateScore("trip", 0, 0, 0.5)]
    assert decoder.decode("trip", point, candidates).decoded_arc_ids == [0]


def test_viterbi_restarts_only_when_no_candidate_can_continue():
    decoder = HMMDecoder(_graph(), DecoderConfig(transition_distance_scale_m=5.0))
    points = [GPSPoint(0, 0, 1, 0), GPSPoint(1, 1, 13, 0), GPSPoint(2, 2, 14, 0)]
    candidates = [
        CandidateScore("trip", 0, 0, 1.0),
        CandidateScore("trip", 1, 1, 0.55),
        CandidateScore("trip", 1, 2, 0.45),
        CandidateScore("trip", 2, 2, 1.0),
    ]
    result = decoder.decode("trip", points, candidates)
    assert result.decoded_arc_ids == [0, 1, 2]
    assert result.expanded_route_segments == [[0, 1], [2]]
    assert [
        warning.gps_point_idx for warning in result.warnings
        if warning.code == "ALL_PREDECESSORS_UNREACHABLE"
    ] == [2]
