from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.emissions import classifier_only_predictions, prepare_emissions
from pathsnap.decoder.schemas import CandidateScore


def test_emissions_deduplicate_clip_and_normalize():
    emissions, warnings = prepare_emissions(
        [
            CandidateScore("trip", 0, 5, 0.2),
            CandidateScore("trip", 0, 5, 0.7),
            CandidateScore("trip", 0, 8, float("nan")),
        ],
        DecoderConfig(),
    )
    states = emissions[0]
    assert [state.arc_id for state in states] == [5, 8]
    assert abs(sum(state.probability for state in states) - 1.0) < 1e-10
    assert classifier_only_predictions(emissions) == {0: 5}
    assert {warning.code for warning in warnings} == {
        "DUPLICATE_CANDIDATE_DEDUPLICATED",
        "INVALID_PROBABILITY_CLIPPED",
    }
