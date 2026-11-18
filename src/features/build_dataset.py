"""Build candidate feature datasets for train (80 trips) and student_test (10 trips)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eda.data_loader import list_trip_ids, load_trip, resolve_data_root
from src.features.candidate_gen import generate_candidates_for_trip
from src.features.feature_engineering import (
    FEATURE_COLS,
    extract_features_for_trip,
    fit_and_apply_scaler,
)


def build_feature_datasets(
    base_dir: Path | str = ".",
    radius_m: float = 50.0,
    max_candidates: int = 20,
    artifacts_dir: Path | str = "artifacts",
    results_dir: Path | str = "results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate candidates, log Candidate Miss Rate, extract features, and fit train scaler."""
    t0 = time.perf_counter()
    data_root = resolve_data_root(base_dir)
    art_dir = Path(artifacts_dir)
    res_dir = Path(results_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    miss_logs: list[dict] = []
    split_frames: dict[str, list[pd.DataFrame]] = {"train": [], "student_test": []}

    for split in ("train", "student_test"):
        split_dir = data_root / split
        for tid in list_trip_ids(split_dir):
            trip = load_trip(split, tid, split_dir)
            if trip.route is None or not len(trip.route):
                raise ValueError(f"Missing reference route for {split}/{tid}")
            tc = generate_candidates_for_trip(
                trip, radius_m=radius_m, max_candidates=max_candidates
            )
            miss_logs.append(
                {
                    "split": split,
                    "trip_id": tid,
                    "radius_m": radius_m,
                    "max_candidates": max_candidates,
                    "n_points": tc.n_points,
                    "n_candidate_rows": len(tc.pt_idx),
                    "avg_cands_per_pt": round(len(tc.pt_idx) / tc.n_points, 3),
                    "n_missed_gt": tc.n_missed_gt,
                    "n_zero_cands": tc.n_zero_cands,
                    "miss_rate_pct": round(100.0 * tc.miss_rate, 4),
                    "zero_cand_rate_pct": round(100.0 * tc.zero_cand_rate, 4),
                }
            )
            feat_df = extract_features_for_trip(tc)
            split_frames[split].append(feat_df)

    miss_df = pd.DataFrame(miss_logs)
    miss_df.to_csv(res_dir / "candidate_miss_rate.csv", index=False)

    train_df = pd.concat(split_frames["train"], ignore_index=True)
    test_df = pd.concat(split_frames["student_test"], ignore_index=True)

    x_train_scaled, x_test_scaled, _ = fit_and_apply_scaler(
        train_df,
        test_df,
        feature_cols=FEATURE_COLS,
        scaler_path=art_dir / "feature_scaler.joblib",
    )

    joblib.dump(
        {
            "train_df": train_df,
            "test_df": test_df,
            "x_train_scaled": x_train_scaled,
            "x_test_scaled": x_test_scaled,
            "feature_cols": FEATURE_COLS,
        },
        art_dir / "feature_datasets.joblib",
        compress=3,
    )

    elapsed = time.perf_counter() - t0
    print(f"Feature dataset build completed in {elapsed:.2f}s (radius={radius_m}m, max_cands={max_candidates})")
    for split in ("train", "student_test"):
        sub = miss_df[miss_df["split"] == split]
        total_pts = int(sub["n_points"].sum())
        total_missed = int(sub["n_missed_gt"].sum())
        total_zero = int(sub["n_zero_cands"].sum())
        total_rows = int(sub["n_candidate_rows"].sum())
        print(
            f"  [{split}] trips={len(sub)} | pts={total_pts:,} | cand_rows={total_rows:,} "
            f"| avg_cands/pt={total_rows / total_pts:.2f} "
            f"| Candidate Miss Rate={100.0 * total_missed / total_pts:.3f}% ({total_missed}/{total_pts}) "
            f"| Zero-Cand Rate={100.0 * total_zero / total_pts:.3f}%"
        )

    return train_df, test_df


if __name__ == "__main__":
    build_feature_datasets()
