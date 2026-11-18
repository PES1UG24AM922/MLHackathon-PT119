"""Measure raw classifier route continuity on the same student-test candidates."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.final_train import Data, SEED, fit_predict, sample_rows
from src.eda.data_loader import load_trip, resolve_data_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/final"))
    args = parser.parse_args()
    selection = json.loads((args.out / "model_selection.json").read_text())
    table = pd.read_csv(args.out / "model_comparison.csv")
    bundle = joblib.load(args.cache)
    names = bundle["feature_cols"]
    train, test = (Data.from_frame(bundle[key], names) for key in ("train_df", "test_df"))
    candidates = bundle["test_df"]["cand_arc_id"].to_numpy(dtype=int)
    del bundle
    root = resolve_data_root(args.data) / "student_test"
    topology = {}
    for trip_id in sorted(set(test.trip)):
        trip = load_trip("student_test", trip_id, root, load_reference_route=False)
        topology[trip_id] = (trip.arcs.to_numpy(dtype=int), np.diff(trip.track.timestamp.to_numpy()))
    fitting = sample_rows(train, np.arange(len(train.y)), selection["fit_row_cap"], SEED)
    starts = np.r_[0, np.flatnonzero((test.trip[1:] != test.trip[:-1])
                                     | (test.point[1:] != test.point[:-1])) + 1]
    counts = np.diff(np.r_[starts, len(test.y)])
    locations = np.arange(len(test.y))
    rows = []
    with threadpool_limits(limits=4):
        for model in table.itertuples():
            fitted, p, _ = fit_predict(model.model, json.loads(model.configuration),
                                       train, fitting, test, locations)
            if model.model == "MLP":
                network = fitted.steps[-1][1]
                pd.DataFrame({
                    "iteration": np.arange(1, len(network.loss_curve_) + 1),
                    "training_loss": network.loss_curve_,
                    "internal_validation_accuracy": network.validation_scores_,
                }).to_csv(args.out / "mlp_learning_history.csv", index=False)
            maxima = np.repeat(np.maximum.reduceat(p, starts), counts)
            best = np.minimum.reduceat(np.where(p == maxima, locations, len(p)), starts)
            if not np.isclose(test.y[best].mean(), model.student_top1, atol=1e-7, rtol=0):
                raise ValueError(f"Reproduced classifier differs for {model.model}")
            trip_keys, point_keys = test.trip[best], test.point[best]
            picked = candidates[best]
            for trip_id, (arcs, gaps) in topology.items():
                point, seq = point_keys[trip_keys == trip_id], picked[trip_keys == trip_id]
                keep = (np.diff(point) == 1) & (gaps[point[:-1]] > 0) & (gaps[point[:-1]] <= 30)
                prev, nxt = seq[:-1][keep], seq[1:][keep]
                continuity = ((prev == nxt) | (arcs[prev, 1] == arcs[nxt, 0])).mean()
                rows.append({"model": model.model, "trip_id": trip_id, "continuity": continuity})
            print("Checked route continuity", model.model, flush=True)
    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(args.out / "model_continuity_by_trip.csv", index=False)
    continuity = breakdown.groupby("model").continuity.mean()
    table["student_route_continuity"] = table.model.map(continuity)
    table.to_csv(args.out / "model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
