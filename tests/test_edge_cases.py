import numpy as np

from pathsnap.decoder import CandidateScore, DecoderConfig, GPSPoint, HMMDecoder, RoadGraph


def _decoder(gap_reset_seconds=5.0):
    return HMMDecoder(
        RoadGraph(np.array([[0, 0], [10, 0]], dtype=float), np.array([[0, 1]], dtype=int)),
        DecoderConfig(gap_reset_seconds=gap_reset_seconds),
    )


def test_empty_candidates_are_unmatched_not_a_crash():
    result = _decoder().decode(
        "trip",
        [GPSPoint(0, 0, 0, 0), GPSPoint(1, 1, 1, 0)],
        [CandidateScore("trip", 0, 0, 1.0)],
    )
    assert result.decoded_arc_ids == [0, None]
    assert any(warning.code == "EMPTY_CANDIDATE_SET" for warning in result.warnings)


def test_large_gap_starts_new_segment():
    result = _decoder().decode(
        "trip",
        [GPSPoint(0, 0, 0, 0), GPSPoint(1, 10, 10, 0)],
        [CandidateScore("trip", 0, 0, 1.0), CandidateScore("trip", 1, 0, 1.0)],
    )
    assert result.decoded_arc_ids == [0, 0]
    assert result.decoded_points[0].segment_id != result.decoded_points[1].segment_id
    assert result.expanded_route_segments == [[0], [0]]
    assert any(warning.code == "LARGE_GPS_GAP" for warning in result.warnings)


def test_unknown_candidate_arc_fails_fast():
    try:
        _decoder().decode("trip", [GPSPoint(0, 0, 0, 0)], [CandidateScore("trip", 0, 99, 1.0)])
    except ValueError as exc:
        assert "unknown arc ID" in str(exc)
    else:
        raise AssertionError("unknown arc must be rejected")


def test_non_increasing_timestamp_restarts_sequence():
    result = _decoder().decode(
        "trip",
        [GPSPoint(0, 2, 0, 0), GPSPoint(1, 1, 1, 0)],
        [CandidateScore("trip", 0, 0, 1.0), CandidateScore("trip", 1, 0, 1.0)],
    )
    assert result.decoded_arc_ids == [0, 0]
    assert any(warning.code == "NON_INCREASING_TIMESTAMP" for warning in result.warnings)


def test_unusable_gps_point_is_preserved_as_unmatched():
    result = _decoder().decode(
        "trip",
        [GPSPoint(0, 0, 0, 0), GPSPoint(1, 1, 1, 0, usable=False)],
        [CandidateScore("trip", 0, 0, 1.0), CandidateScore("trip", 1, 0, 1.0)],
    )
    assert result.decoded_arc_ids == [0, None]
    assert any(warning.code == "UNUSABLE_GPS_POINT" for warning in result.warnings)


def test_all_disconnected_predecessors_start_a_new_segment():
    graph = RoadGraph(
        np.array([[0, 0], [10, 0], [0, 20], [10, 20]], dtype=float),
        np.array([[0, 1], [2, 3]], dtype=int),
    )
    result = HMMDecoder(graph).decode(
        "trip",
        [GPSPoint(0, 0, 0, 0), GPSPoint(1, 1, 10, 20)],
        [CandidateScore("trip", 0, 0, 1.0), CandidateScore("trip", 1, 1, 1.0)],
    )
    assert result.decoded_arc_ids == [0, 1]
    assert result.decoded_points[0].segment_id != result.decoded_points[1].segment_id
    assert any(warning.code == "ALL_PREDECESSORS_UNREACHABLE" for warning in result.warnings)


def test_candidate_trip_mismatch_is_rejected():
    try:
        _decoder().decode("trip_a", [GPSPoint(0, 0, 0, 0)], [CandidateScore("trip_b", 0, 0, 1.0)])
    except ValueError as exc:
        assert "trip_id" in str(exc)
    else:
        raise AssertionError("cross-trip candidates must be rejected")
