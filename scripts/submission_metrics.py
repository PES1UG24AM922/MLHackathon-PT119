"""Reproduce the student-only model diagnostics used in the technical report."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.final_train import (
    DEFAULTS, SEED, Data, estimator, observation_starts, sample_rows, score, top_candidates,
)
from src.eda.data_loader import load_trip, resolve_data_root


def build_estimator(name, params):
    if name == "KNeighbors":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(algorithm="brute", n_jobs=4, **params))
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_jobs=4, random_state=SEED, **params)
    return estimator(name, params)


def fit_model(name, params, train, rows):
    model = build_estimator(name, params)
    return model.fit(train.x[rows], train.y[rows])


def observation_metrics(frame, probabilities, topology):
    trips = frame.trip_id.to_numpy()
    points = frame.pt_idx.to_numpy()
    starts = observation_starts(trips, points)
    picks = top_candidates(probabilities, starts)
    targets = frame.gt_arc_id.to_numpy()[starts]
    predicted = frame.cand_arc_id.to_numpy()[picks]
    true_mass = np.add.reduceat(probabilities * frame.is_true_arc.to_numpy(), starts)
    total_mass = np.add.reduceat(probabilities, starts)
    true_probability = np.divide(true_mass, total_mass, out=np.zeros(len(starts)), where=total_mass > 0)
    # A model can assign zero to every candidate (e.g., KNN). Use a uniform
    # distribution over retrieved arcs. A missed target has zero probability.
    counts = np.diff(np.r_[starts, len(frame)])
    empty = total_mass == 0
    target_present = np.add.reduceat(frame.is_true_arc.to_numpy(), starts)
    true_probability[empty] = target_present[empty] / counts[empty]
    losses = -np.log(np.clip(true_probability, 1e-15, 1.0))
    trip_ids, point_ids = trips[starts], points[starts]
    rows = []
    for trip_id, (arcs, gaps) in topology.items():
        mask = trip_ids == trip_id
        point, seq = point_ids[mask], predicted[mask]
        if not np.array_equal(point, np.arange(len(gaps) + 1)):
            raise ValueError(f"Student-test candidate cache does not cover every observation for {trip_id}")
        keep = (gaps > 0) & (gaps <= 30)
        previous, following = seq[:-1][keep], seq[1:][keep]
        connected = (previous == following) | (arcs[previous, 1] == arcs[following, 0])
        rows.append({
            "trip_id": trip_id,
            "observations": len(point),
            "top1": float(np.mean(seq == targets[mask])),
            "macro_f1": float(f1_score(targets[mask], seq, average="macro", zero_division=0)),
            "log_loss": float(losses[mask].mean()),
            "continuity": float(connected.mean()) if len(connected) else np.nan,
        })
    breakdown = pd.DataFrame(rows)
    return breakdown, {
        "student_top1": float(np.mean(predicted == targets)),
        "student_obs_macro_f1": float(breakdown.macro_f1.mean()),
        "student_obs_log_loss": float(losses.mean()),
        "student_route_continuity": float(breakdown.continuity.mean()),
    }


def student_diagnostics(model, test, frame, topology):
    started = time.perf_counter()
    probabilities = model.predict_proba(test.x)[:, 1]
    seconds = time.perf_counter() - started
    by_trip, results = observation_metrics(frame, probabilities, topology)
    candidate = score(test, np.arange(len(test.y)), probabilities)
    if not np.isclose(candidate["top1"], results["student_top1"]):
        raise ValueError("Candidate and observed-arc Top-1 disagree")
    tn, fp, fn, tp = confusion_matrix(test.y, probabilities >= .5, labels=[0, 1]).ravel()
    return results | {
        "student_candidate_macro_f1": candidate["macro_f1"],
        "student_candidate_log_loss": candidate["log_loss"],
        "student_inference_s": seconds,
    }, by_trip, dict(tn=tn, fp=fp, fn=fn, tp=tp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", default=Path("results/final"), type=Path)
    args = parser.parse_args()
    selection = json.loads(Path("results/final/model_selection.json").read_text())
    bundle = joblib.load(args.cache)
    features = bundle["feature_cols"]
    train, test = (Data.from_frame(bundle[key], features) for key in ("train_df", "test_df"))
    test_frame = bundle["test_df"][["trip_id", "pt_idx", "gt_arc_id", "cand_arc_id", "is_true_arc"]]
    del bundle
    if len(set(train.trip)) != 80 or len(set(test.trip)) != 10 or set(train.trip) & set(test.trip):
        raise ValueError("Expected the separate 80-trip training and 10-trip student-test partitions")

    root = resolve_data_root(args.data) / "student_test"
    topology = {}
    for trip_id in sorted(set(test.trip)):
        trip = load_trip("student_test", trip_id, root, load_reference_route=False)
        topology[trip_id] = (trip.arcs.to_numpy(dtype=int), np.diff(trip.track.timestamp.to_numpy()))
    folds = list(GroupKFold(5).split(train.x, train.y, groups=train.trip))
    all_rows = np.arange(len(train.y))
    training_check = np.flatnonzero(np.isin(train.trip, sorted(set(train.trip))[:10]))
    cap = selection["fit_row_cap"]
    specs = [(name, name, params, cap) for name, params in DEFAULTS.items()]
    specs.extend([
        ("KNeighbors", "KNeighbors", {"n_neighbors": 15, "weights": "distance"}, 10_000),
        ("ExtraTrees", "ExtraTrees", {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 5,
                                     "bootstrap": True}, cap),
        ("Tuned raw", selection["winner"], selection["best_params"], cap),
    ])
    comparisons, fold_rows, breakdowns, matrices = [], [], [], []
    with threadpool_limits(limits=4):
        for label, name, params, fit_cap in specs:
            print(f"Evaluating {label}", flush=True)
            validation_rows = []
            for fold, (fit, held_out) in enumerate(folds, start=1):
                model = fit_model(name, params, train, sample_rows(train, fit, fit_cap, SEED + fold - 1))
                estimates = model.predict_proba(train.x[held_out])[:, 1]
                validation_rows.append({"model": label, "fold": fold, **score(train, held_out, estimates)})
                print(f"  fold {fold}/5", flush=True)
            fold_rows.extend(validation_rows)
            model = fit_model(name, params, train, sample_rows(train, all_rows, fit_cap, SEED))
            observed, breakdown, matrix = student_diagnostics(model, test, test_frame, topology)
            fit_score = score(train, training_check, model.predict_proba(train.x[training_check])[:, 1])
            cv = pd.DataFrame(validation_rows).drop(columns=["model", "fold"]).mean().add_prefix("cv_")
            comparisons.append({"model": label, "fit_cap": fit_cap, "configuration": json.dumps(params),
                                "fit_pool_top1": fit_score["top1"], **cv.to_dict(), **observed})
            breakdowns.append(breakdown.assign(model=label))
            matrices.append({"model": label, **matrix})
        released = joblib.load("artifacts/final_emission.joblib")["model"]
        observed, breakdown, matrix = student_diagnostics(released, test, test_frame, topology)
        comparisons.append({"model": "Tuned sigmoid", "fit_cap": cap,
                            "configuration": json.dumps(selection["best_params"]), **observed})
        breakdowns.append(breakdown.assign(model="Tuned sigmoid"))
        matrices.append({"model": "Tuned sigmoid", **matrix})

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparisons).to_csv(args.out / "student_model_diagnostics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out / "diagnostics_folds.csv", index=False)
    pd.concat(breakdowns, ignore_index=True).to_csv(args.out / "student_diagnostics_by_trip.csv", index=False)
    pd.DataFrame(matrices).to_csv(args.out / "candidate_confusion_counts.csv", index=False)


if __name__ == "__main__":
    main()
