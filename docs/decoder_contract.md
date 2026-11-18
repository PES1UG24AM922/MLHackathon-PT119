# Person 3 Decoder Contract

## Boundary

The decoder is an inference component. Its inputs are GPS observations,
directed graph artifacts, candidate arc IDs, and calibrated probabilities. It
does not accept route labels or reference geometry.

Offline scripts may separately load `reference_labels.csv` to compare
classifier-only and HMM/Viterbi predictions. This boundary prevents evaluator
labels from entering the prediction pipeline or the UI.

## Required validations

1. GPS indices are unique and ordered.
2. Candidate rows belong to one requested trip.
3. Candidate GPS indices exist.
4. Candidate arc IDs exist in the directed graph.
5. Nodes use projected metre coordinates.
6. Model and decoder manifests use compatible schema versions.

## API-safe result fields

`trip_id`, predicted point-level arc IDs, compact/expanded predicted routes,
warnings, decoder version, and inference time are safe result fields. Reference
arc IDs, reference-route geometry, and per-point correctness labels are not.

## Warning codes

- `EMPTY_CANDIDATE_SET`
- `UNUSABLE_GPS_POINT`
- `LARGE_GPS_GAP`
- `NON_INCREASING_TIMESTAMP`
- `DUPLICATE_CANDIDATE_DEDUPLICATED`
- `INVALID_PROBABILITY_CLIPPED`
- `ALL_PREDECESSORS_DISCONNECTED`
- `DISPLAY_ROUTE_DISCONNECTED`

Person 4 should expose a safe summary of these codes in API logs and UI status,
not raw tracebacks.
