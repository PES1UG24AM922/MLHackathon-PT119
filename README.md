# PathSnap

This repository contains the team implementation for PathSnap.  The current
module is Person 3's directed HMM/Viterbi route decoder.  It is intentionally
separated into a label-free inference package and offline evaluation scripts.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,visualization]'
pytest -q
```

The test runner prints a one-line explanation after every test, for example
`PASS: resets the HMM after a large GPS gap`.

## Person 3 deliverables

- `src/pathsnap/decoder/`: directed graph, calibrated emissions, transition
  scoring, segmented Viterbi decoding, post-decoding route expansion, and
  offline metrics.
- `configs/hmm.yaml`: versioned decoder settings. It uses JSON syntax, which
  is valid YAML, so no YAML dependency is required.
- `scripts/decode_trip.py`: label-free inference for one trip.
- `scripts/tune_decoder.py`: train-partition-only five-fold decoder tuning.
- `scripts/evaluate_classifier_vs_hmm.py`: offline classifier-only versus HMM
  comparison using reference labels.
- `scripts/generate_decoder_map.py`: real Folium comparison-map output.
- `tests/`: unit and edge-case regression tests.

## Data contract

Person 1 should export projected CSV artifacts. Coordinates used by graph and
transition calculations must be metres, not raw longitude/latitude degrees.

`gps.csv`:

```text
gps_point_idx,timestamp_s,x_m,y_m,longitude,latitude,usable
```

`nodes.csv`:

```text
x_m,y_m,longitude,latitude
```

`arcs.csv`:

```text
from_node_id,to_node_id
```

`candidates.csv` from Person 2:

```text
trip_id,gps_point_idx,candidate_arc_id,calibrated_probability,candidate_miss
```

`reference_labels.csv` is **offline evaluation only**:

```text
gps_point_idx,reference_arc_id
```

The API/inference adapter must never provide the final decoder with
`reference_labels.csv` or raw `.route` data.

## Offline manifest

All scripts use a JSON manifest so code can be run on a local machine or on a
different laptop without hard-coded paths. Paths are relative to the manifest.

```json
{
  "trips": [
    {
      "trip_id": "00000001",
      "partition": "train",
      "gps": "processed/train/00000001/gps.csv",
      "nodes": "processed/train/00000001/nodes.csv",
      "arcs": "processed/train/00000001/arcs.csv",
      "candidates": "scores/oof_train_00000001.csv",
      "reference_labels": "labels/train_00000001.csv"
    }
  ]
}
```

For decoder tuning, training scores must be out-of-fold predictions.  Do not
tune decoder parameters against in-sample model probabilities.

## Commands

```bash
# Unit and edge-case checks
pytest -q

# Label-free inference: safe for Person 4's backend adapter
python scripts/decode_trip.py --manifest data_manifest.json --trip-id 00000001 \
  --output results/predictions/00000001.csv

# Tune only on the 80 training trips, then save trial history + best config
python scripts/tune_decoder.py --manifest data_manifest.json \
  --grid configs/hmm_grid.json

# Run the compulsory classifier-only versus HMM comparison offline
python scripts/evaluate_classifier_vs_hmm.py --manifest data_manifest.json \
  --partition student_test

# Generate a report map; reference layer is only valid for offline evaluation
python scripts/generate_decoder_map.py --manifest data_manifest.json \
  --trip-id 00000001 --include-reference \
  --output results/maps/00000001_hmm.html
```

## Decoder behaviour

- States are directed candidate arcs.
- Emissions are Person 2's calibrated per-candidate probabilities, normalized
  within each GPS observation and decoded in log space.
- Transitions use directed shortest paths, GPS displacement, time interval,
  network distance, and plausible speed.
- Large GPS gaps, unusable GPS, empty candidates, and non-increasing
  timestamps split/reset the sequence instead of forcing an invalid route.
- Invalid candidate probabilities are clipped and logged; duplicate candidates
  are deterministically deduplicated.
- The compact map route is not used for point-level metrics.

## Integration with Person 4

Load `RoadGraph`, `DecoderConfig`, and `HMMDecoder` at API startup. For each
request call:

```python
result = decoder.decode(trip_id, gps_points, candidate_scores)
```

Return predicted route geometry, per-point predicted arc IDs when appropriate,
warnings, decoder version, and elapsed time. Evaluate against reference labels
only in a separate server-side evaluator; never return those labels to the UI.
