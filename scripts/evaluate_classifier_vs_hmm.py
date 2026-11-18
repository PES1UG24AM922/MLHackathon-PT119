"""Offline only: compare classifier-only and HMM predictions against labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.decoder import DecoderConfig, HMMDecoder  # noqa: E402
from pathsnap.decoder.emissions import classifier_only_predictions, prepare_emissions  # noqa: E402
from pathsnap.decoder.evaluation import candidate_miss_rate, evaluate_predictions, result_prediction_map  # noqa: E402
from pathsnap.decoder.experiments import load_offline_trip  # noqa: E402
from pathsnap.decoder.io import read_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline classifier-vs-HMM evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--partition", default="student_test")
    parser.add_argument("--config", default=str(ROOT / "configs" / "hmm.yaml"))
    parser.add_argument("--output", default=str(ROOT / "results" / "classifier_vs_hmm.csv"))
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = read_manifest(manifest_path)
    config = DecoderConfig.from_file(args.config)
    rows = []
    for entry in manifest["trips"]:
        if entry["partition"] != args.partition:
            continue
        trip = load_offline_trip(entry, manifest_path.parent, require_labels=True)
        emissions, _ = prepare_emissions(trip.candidates, config)
        flags = {}
        for candidate in trip.candidates:
            flags[candidate.gps_point_idx] = flags.get(candidate.gps_point_idx, False) or candidate.candidate_miss
        miss_rate = candidate_miss_rate(flags.values())
        baseline = classifier_only_predictions(emissions)
        baseline_metrics = evaluate_predictions(
            baseline, trip.reference_labels or {}, trip.graph, trip.gps_points, candidate_miss_rate=miss_rate
        )
        result = HMMDecoder(trip.graph, config).decode(trip.trip_id, trip.gps_points, trip.candidates)
        hmm_metrics = evaluate_predictions(
            result_prediction_map(result), trip.reference_labels or {}, trip.graph, trip.gps_points, candidate_miss_rate=miss_rate
        )
        for method, metrics in (("classifier_only", baseline_metrics), ("hmm_viterbi", hmm_metrics)):
            rows.append({
                "trip_id": trip.trip_id,
                "partition": trip.partition,
                "method": method,
                **metrics,
                "inference_time_ms": result.inference_time_ms if method == "hmm_viterbi" else 0.0,
                "warning_count": len(result.warnings) if method == "hmm_viterbi" else 0,
            })
    if not rows:
        raise SystemExit(f"No {args.partition!r} trips found in manifest.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} method/trip evaluations to {output}")


if __name__ == "__main__":
    main()
