"""Train-partition-only grid tuning for reproducible HMM parameters."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import itertools
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.decoder import DecoderConfig, HMMDecoder  # noqa: E402
from pathsnap.decoder.evaluation import (  # noqa: E402
    evaluate_predictions,
    final_candidate_miss_rate,
    result_prediction_map,
    result_segment_map,
)
from pathsnap.decoder.experiments import load_offline_trip  # noqa: E402
from pathsnap.decoder.io import read_manifest  # noqa: E402


def parameter_sets(base: DecoderConfig, grid: dict[str, list[object]]):
    invalid = set(grid) - set(base.to_dict())
    if invalid:
        raise ValueError(f"grid includes unknown config fields: {sorted(invalid)}")
    names = sorted(grid)
    values = [grid[name] for name in names]
    for combination in itertools.product(*values):
        yield replace(base, **dict(zip(names, combination)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune decoder on train trips only.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "hmm.yaml"))
    parser.add_argument("--grid", required=True, help="JSON file mapping config parameter to candidate values")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default=str(ROOT / "results" / "decoder_tuning_trials.csv"))
    parser.add_argument("--best-config-output", default=str(ROOT / "results" / "best_hmm_config.json"))
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = read_manifest(manifest_path)
    entries = [entry for entry in manifest["trips"] if entry["partition"] == "train"]
    if len(entries) < args.folds:
        raise SystemExit("Need at least as many train trips as folds.")
    with Path(args.grid).open(encoding="utf-8") as handle:
        grid = json.load(handle)
    base = DecoderConfig.from_file(args.config)
    rows = []
    for trial_id, config in enumerate(parameter_sets(base, grid), start=1):
        fold_scores: list[float] = []
        for fold in range(args.folds):
            validation_entries = [entry for index, entry in enumerate(entries) if index % args.folds == fold]
            metrics_for_fold = []
            for entry in validation_entries:
                trip = load_offline_trip(entry, manifest_path.parent, require_labels=True)
                result = HMMDecoder(trip.graph, config).decode(trip.trip_id, trip.gps_points, trip.candidates)
                metrics_for_fold.append(
                    evaluate_predictions(
                        result_prediction_map(result),
                        trip.reference_labels or {},
                        trip.graph,
                        trip.gps_points,
                        candidate_miss_rate=final_candidate_miss_rate(
                            trip.gps_points, trip.candidates, trip.reference_labels or {}
                        ),
                        segment_ids=result_segment_map(result),
                    )
                )
            macro_f1 = statistics.fmean(metric["macro_f1"] for metric in metrics_for_fold)
            top1 = statistics.fmean(metric["top1_accuracy"] for metric in metrics_for_fold)
            fold_scores.append(macro_f1)
            rows.append({
                "trial_id": trial_id,
                "fold": fold,
                "cv_macro_f1": macro_f1,
                "cv_top1_accuracy": top1,
                "config_json": json.dumps(config.to_dict(), sort_keys=True),
            })
        print(f"trial={trial_id} mean_macro_f1={statistics.fmean(fold_scores):.5f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["trial_id"]), []).append(row)
    best_trial_id = max(
        grouped,
        key=lambda trial_id: statistics.fmean(float(row["cv_macro_f1"]) for row in grouped[trial_id]),
    )
    best_config = json.loads(str(grouped[best_trial_id][0]["config_json"]))
    Path(args.best_config_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.best_config_output).write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    print(f"Best train-CV trial: {best_trial_id}; config saved to {args.best_config_output}")


if __name__ == "__main__":
    main()
