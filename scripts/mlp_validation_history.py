"""Record an MLP learning curve on a small, trip-separated training diagnostic."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.eda.data_loader import list_trip_ids, load_trip, resolve_data_root
from src.features.candidate_gen import generate_candidates_for_trip
from src.features.feature_engineering import FEATURE_COLS, extract_features_for_trip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/final/mlp_grouped_loss.csv"))
    args = parser.parse_args()
    root = resolve_data_root(args.data) / "train"
    trip_ids = list_trip_ids(root)[:12]
    if len(trip_ids) != 12:
        parser.error("the diagnostic requires twelve training trips")
    fitting, validation = trip_ids[:9], trip_ids[9:]

    def load_features(ids):
        frames = []
        for trip_id in ids:
            trip = load_trip("train", trip_id, root)
            frames.append(extract_features_for_trip(generate_candidates_for_trip(trip)))
        return pd.concat(frames, ignore_index=True)

    train, held_out = load_features(fitting), load_features(validation)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLS].to_numpy(dtype=np.float32))
    x_validation = scaler.transform(held_out[FEATURE_COLS].to_numpy(dtype=np.float32))
    y_train = train["is_true_arc"].to_numpy()
    y_validation = held_out["is_true_arc"].to_numpy()
    model = MLPClassifier(hidden_layer_sizes=(24,), alpha=0.0001,
                          learning_rate_init=0.003, batch_size=512, random_state=42)
    history = []
    with threadpool_limits(limits=4):
        for epoch in range(1, 56):
            model.partial_fit(x_train, y_train, classes=[0, 1])
            history.append({
                "epoch": epoch,
                "training_log_loss": log_loss(y_train, model.predict_proba(x_train), labels=[0, 1]),
                "validation_log_loss": log_loss(y_validation, model.predict_proba(x_validation), labels=[0, 1]),
                "fitting_trip_ids": " ".join(fitting),
                "validation_trip_ids": " ".join(validation),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    print(f"Saved {len(history)} epochs to {args.output}")


if __name__ == "__main__":
    main()
