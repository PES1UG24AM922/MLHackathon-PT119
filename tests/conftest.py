"""Readable terminal descriptions for the decoder acceptance suite."""

from __future__ import annotations

import pytest


TEST_PURPOSES = {
    "test_directed_shortest_path_and_parallel_arcs": "checks directed routing, parallel-arc tie breaking, and self-loops",
    "test_graph_rejects_invalid_endpoint": "rejects arcs with nonexistent node IDs",
    "test_emissions_deduplicate_clip_and_normalize": "clips invalid scores, deduplicates candidates, and normalizes emissions",
    "test_viterbi_prefers_connected_route_over_high_disconnected_emission": "prefers a topologically valid route over a disconnected high-score candidate",
    "test_tie_break_is_deterministic": "uses the lowest arc ID for equal-score ties",
    "test_empty_candidates_are_unmatched_not_a_crash": "leaves a GPS point unmatched when no candidates exist",
    "test_large_gap_starts_new_segment": "resets the HMM after a large GPS gap",
    "test_unknown_candidate_arc_fails_fast": "rejects candidates that do not exist in the road graph",
    "test_non_increasing_timestamp_restarts_sequence": "resets safely on duplicate or backwards timestamps",
    "test_unusable_gps_point_is_preserved_as_unmatched": "preserves preprocessing-marked invalid GPS points without decoding them",
    "test_all_disconnected_predecessors_start_a_new_segment": "starts a new route segment when every predecessor is unreachable",
    "test_candidate_trip_mismatch_is_rejected": "prevents candidate scores from another trip entering the decode call",
    "test_unordered_or_duplicate_gps_indices_are_rejected": "enforces unique, ordered GPS indices",
    "test_candidate_must_reference_an_existing_gps_point": "rejects candidate rows for nonexistent GPS observations",
    "test_mixed_trip_emissions_are_rejected": "rejects mixed-trip probability groups",
    "test_invalid_configuration_is_rejected": "rejects unsafe HMM configuration values",
    "test_projected_arc_offsets_prevent_long_arc_junction_error": "uses arc projection offsets at a long-arc junction",
    "test_metrics_use_point_level_predictions": "computes accuracy, Macro F1, and continuity from point-level states",
    "test_macro_f1_handles_missed_class": "keeps Macro F1 correct when a road-arc class is missed",
    "test_unmatched_eligible_point_counts_against_accuracy_and_coverage": "counts abstentions as misses and reports prediction coverage",
    "test_candidate_csv_preserves_zero_padded_trip_id": "preserves leading zeroes in real-format trip IDs",
    "test_label_free_candidate_and_feature_generation_needs_no_route": "builds inference candidates/features when no reference route exists",
    "test_offline_scripts_execute_end_to_end": "runs decode, label-isolation, offline comparison, and five-fold tuning scripts on synthetic artifacts",
    "test_person1_person2_adapter_decodes_label_free_frames": "connects Person 1 feature frames and a calibrated Person 2 model to label-free HMM decoding",
    "test_adapter_requires_calibration_and_matching_feature_schema": "rejects uncalibrated or schema-incompatible model artifacts",
}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Print one concise explanation after each test's call phase."""
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    reporter = item.config.pluginmanager.getplugin("terminalreporter")
    if reporter is None:
        return
    purpose = TEST_PURPOSES.get(item.name, item.name.replace("_", " "))
    prefix = "PASS" if report.passed else "FAIL"
    reporter.write_line(f"{prefix}: {purpose}", green=report.passed, red=report.failed)
