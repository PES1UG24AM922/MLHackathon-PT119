import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_trip(root: Path, trip_id: str, partition: str) -> dict[str, str]:
    pd.DataFrame(
        {
            "gps_point_idx": [0, 1],
            "timestamp_s": [0.0, 1.0],
            "x_m": [1.0, 11.0],
            "y_m": [0.0, 0.0],
            "longitude": [8.0, 8.0001],
            "latitude": [49.0, 49.0],
            "usable": [True, True],
        }
    ).to_csv(root / f"{trip_id}_gps.csv", index=False)
    pd.DataFrame(
        {
            "x_m": [0.0, 10.0, 20.0],
            "y_m": [0.0, 0.0, 0.0],
            "longitude": [8.0, 8.0001, 8.0002],
            "latitude": [49.0, 49.0, 49.0],
        }
    ).to_csv(root / f"{trip_id}_nodes.csv", index=False)
    pd.DataFrame({"from_node_id": [0, 1], "to_node_id": [1, 2]}).to_csv(
        root / f"{trip_id}_arcs.csv", index=False
    )
    pd.DataFrame(
        {
            "trip_id": [trip_id, trip_id, trip_id, trip_id],
            "gps_point_idx": [0, 0, 1, 1],
            "candidate_arc_id": [0, 1, 0, 1],
            "calibrated_probability": [0.9, 0.1, 0.1, 0.9],
            "candidate_miss": [False, False, False, False],
        }
    ).to_csv(root / f"{trip_id}_candidates.csv", index=False)
    pd.DataFrame({"gps_point_idx": [0, 1], "reference_arc_id": [0, 1]}).to_csv(
        root / f"{trip_id}_labels.csv", index=False
    )
    return {
        "trip_id": trip_id,
        "partition": partition,
        "gps": f"{trip_id}_gps.csv",
        "nodes": f"{trip_id}_nodes.csv",
        "arcs": f"{trip_id}_arcs.csv",
        "candidates": f"{trip_id}_candidates.csv",
        "reference_labels": f"{trip_id}_labels.csv",
    }


def test_offline_scripts_execute_end_to_end(tmp_path: Path):
    entries = [_write_trip(tmp_path, f"train_{index}", "train") for index in range(5)]
    entries.append(_write_trip(tmp_path, "student_0", "student_test"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"trips": entries}), encoding="utf-8")
    grid = tmp_path / "grid.json"
    grid.write_text(json.dumps({"emission_weight": [1.0]}), encoding="utf-8")

    def run(script: str, *arguments: str):
        return subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script), *arguments],
            check=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

    predictions = tmp_path / "predictions.csv"
    run("decode_trip.py", "--manifest", str(manifest), "--trip-id", "train_0", "--output", str(predictions))
    assert predictions.exists()
    evaluation = tmp_path / "comparison.csv"
    run("evaluate_classifier_vs_hmm.py", "--manifest", str(manifest), "--output", str(evaluation))
    assert len(pd.read_csv(evaluation)) == 2
    trials = tmp_path / "trials.csv"
    best = tmp_path / "best.json"
    run(
        "tune_decoder.py",
        "--manifest",
        str(manifest),
        "--grid",
        str(grid),
        "--output",
        str(trials),
        "--best-config-output",
        str(best),
    )
    assert len(pd.read_csv(trials)) == 5
    assert json.loads(best.read_text(encoding="utf-8"))["emission_weight"] == 1.0
