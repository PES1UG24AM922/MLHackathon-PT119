"""Compare, tune, and calibrate emissions on the organizer's student data."""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.models.calibration import SigmoidScorer, fit_sigmoid, log_odds

SEED = 42
FEATURE_VERSION = "pathsnap-features-v2-ordered-labels"
STARTUP_TRIALS = 4
MODEL_NAMES = ["GaussianNB", "LogisticRegression", "RandomForest", "HistGradientBoosting", "MLP"]
DEFAULTS = {
    "GaussianNB": {"var_smoothing": 1e-9},
    "LogisticRegression": {"C": 1.0},
    "RandomForest": {"n_estimators": 48, "max_depth": 13, "min_samples_leaf": 8, "max_features": 0.8},
    "HistGradientBoosting": {"max_iter": 70, "max_leaf_nodes": 23, "learning_rate": 0.12, "l2_regularization": 0.3},
    "MLP": {"alpha": 0.0001, "learning_rate_init": 0.003, "width": 24},
}
SEARCH = {
    "GaussianNB": {"var_smoothing": ("log", 1e-13, 1e-4)},
    "LogisticRegression": {"C": ("log", 0.005, 20.0)},
    "RandomForest": {"n_estimators": ("int", 28, 72), "max_depth": ("int", 8, 18),
                     "min_samples_leaf": ("int", 3, 22), "max_features": ("float", 0.45, 1.0)},
    "HistGradientBoosting": {"max_iter": ("int", 40, 110), "max_leaf_nodes": ("int", 11, 47),
                             "learning_rate": ("log", 0.035, 0.24), "l2_regularization": ("log", 0.005, 4.0)},
    "MLP": {"alpha": ("log", 1e-6, 0.03), "learning_rate_init": ("log", 0.0007, 0.014), "width": ("int", 12, 42)},
}


@dataclass
class Data:
    x: np.ndarray
    y: np.ndarray
    trip: np.ndarray
    point: np.ndarray

    @classmethod
    def from_frame(cls, frame, features):
        return cls(frame[features].to_numpy(dtype=np.float32),
                   frame["is_true_arc"].to_numpy(dtype=np.int8),
                   frame["trip_id"].astype(str).to_numpy(),
                   frame["pt_idx"].to_numpy(dtype=np.int32))


def estimator(name, params):
    if name == "GaussianNB":
        return GaussianNB(**params)
    if name == "LogisticRegression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=180, random_state=SEED, **params))
    if name == "RandomForest":
        return RandomForestClassifier(n_jobs=4, random_state=SEED, **params)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **params)
    options = dict(params)
    width = options.pop("width")
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(width,), max_iter=55, batch_size=512,
        early_stopping=True, n_iter_no_change=6, random_state=SEED, **options
    ))


def sample_rows(data, rows, cap, seed):
    if len(rows) <= cap:
        return rows
    sampling = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=seed)
    keep, _ = next(sampling.split(np.zeros(len(rows), dtype=np.int8), data.y[rows]))
    return rows[keep]


def score(data, rows, probabilities):
    yt = data.y[rows]
    trips, points = data.trip[rows], data.point[rows]
    starts = np.r_[0, np.flatnonzero((trips[1:] != trips[:-1]) | (points[1:] != points[:-1])) + 1]
    counts = np.diff(np.r_[starts, len(yt)])
    maximum = np.repeat(np.maximum.reduceat(probabilities, starts), counts)
    locations = np.arange(len(yt))
    best = np.minimum.reduceat(np.where(probabilities == maximum, locations, len(yt)), starts)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yt, probabilities >= 0.5, average="macro", zero_division=0
    )
    return dict(top1=float(yt[best].mean()), macro_f1=f1, macro_precision=precision,
                macro_recall=recall, log_loss=log_loss(yt, probabilities, labels=[0, 1]))


def fit_predict(name, params, train, fitting, evaluation, requested):
    model = estimator(name, params)
    started = time.perf_counter()
    model.fit(train.x[fitting], train.y[fitting])
    fit_time = time.perf_counter() - started
    started = time.perf_counter()
    pred = model.predict_proba(evaluation.x[requested])[:, 1]
    return model, pred, dict(train_s=fit_time, inference_s=time.perf_counter() - started)


def validation(name, params, train, folds, cap):
    output = []
    for i, (fitting, validation_rows) in enumerate(folds):
        fit_rows = sample_rows(train, fitting, cap, SEED + i)
        _, probabilities, timing = fit_predict(name, params, train, fit_rows, train, validation_rows)
        output.append({"fold": i + 1, **score(train, validation_rows, probabilities), **timing})
    return pd.DataFrame(output)


def search_params(name, trial):
    params = {}
    for key, (kind, lo, hi) in SEARCH[name].items():
        if kind == "int":
            params[key] = trial.suggest_int(key, lo, hi)
        else:
            params[key] = trial.suggest_float(key, lo, hi, log=(kind == "log"))
    return params


def calibrate(name, params, train, folds, cap):
    """Train a sigmoid on validation scores without exposing decoder tuning trips."""
    raw = np.empty(len(train.y))
    calibrated = np.empty(len(train.y), dtype=np.float32)
    fold_ids = np.empty(len(train.y), dtype=np.int8)
    for i, (fitting, held_out) in enumerate(folds):
        fit_rows = sample_rows(train, fitting, cap, SEED + i)
        _, raw[held_out], _ = fit_predict(name, params, train, fit_rows, train, held_out)
        inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED + i)
        subtrain, calibrate_rows = next(inner.split(fitting, groups=train.trip[fitting]))
        cal_fit_rows = sample_rows(train, fitting[subtrain], cap, SEED + i)
        cal_rows = fitting[calibrate_rows]
        model, cal_scores, _ = fit_predict(name, params, train, cal_fit_rows, train, cal_rows)
        fold_calibrator = fit_sigmoid(cal_scores, train.y[cal_rows])
        calibrated[held_out] = fold_calibrator.predict_proba(
            log_odds(model.predict_proba(train.x[held_out])[:, 1])
        )[:, 1]
        fold_ids[held_out] = i + 1
    return fit_sigmoid(raw, train.y), raw, calibrated, fold_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/final"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/final_emission.joblib"))
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--cap", type=int, default=100_000)
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    if args.trials <= STARTUP_TRIALS:
        parser.error(f"use more than {STARTUP_TRIALS} trials so TPE can propose trials")
    args.out.mkdir(parents=True, exist_ok=True)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    cache = joblib.load(args.cache)
    features = list(cache["feature_cols"])
    train = Data.from_frame(cache["train_df"], features)
    test = Data.from_frame(cache["test_df"], features)
    assert len(set(train.trip)) == 80 and len(set(test.trip)) == 10
    assert set(train.trip).isdisjoint(test.trip)
    folds = list(GroupKFold(5).split(train.x, train.y, groups=train.trip))
    full_train = sample_rows(train, np.arange(len(train.y)), args.cap, SEED)
    all_test = np.arange(len(test.y))
    del cache

    fold_rows, model_rows = [], []
    with threadpool_limits(limits=4):
        for name in MODEL_NAMES:
            cfg = DEFAULTS[name]
            cv = validation(name, cfg, train, folds, args.cap)
            fold_rows.append(cv.assign(model=name))
            model_rows.append({"model": name, **cv.mean(numeric_only=True).drop("fold").add_prefix("cv_").to_dict()})
            print(name, model_rows[-1]["cv_top1"], flush=True)
        pd.concat(fold_rows).to_csv(args.out / "model_folds.csv", index=False)
        winner = max(model_rows, key=lambda row: (row["cv_top1"], -row["cv_log_loss"]))["model"]
        sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=STARTUP_TRIALS)
        study = optuna.create_study(sampler=sampler, direction="maximize")

        def objective(trial):
            cv = validation(winner, search_params(winner, trial), train, folds, args.cap)
            trial.set_user_attr("fold_top1", cv.top1.to_list())
            trial.set_user_attr("cv_macro_f1", float(cv.macro_f1.mean()))
            trial.set_user_attr("cv_log_loss", float(cv.log_loss.mean()))
            return float(cv.top1.mean())

        study.optimize(objective, n_trials=args.trials, callbacks=[
            lambda current, _: current.trials_dataframe().to_csv(args.out / "bayesian_trials.csv", index=False)
        ])
        best = study.best_params
        plan = {"seed": SEED, "winner": winner, "baseline_params": DEFAULTS[winner],
                "search_space": SEARCH[winner], "best_params": best, "objective": "mean five-fold train Top-1",
                "best_cv_top1": study.best_value, "startup_trials": STARTUP_TRIALS,
                "trials": args.trials, "fit_row_cap": args.cap, "calibration": "sigmoid on training OOF probabilities"}
        (args.out / "model_selection.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

        for row in model_rows:
            name = row["model"]
            _, pred, timing = fit_predict(name, DEFAULTS[name], train, full_train, test, all_test)
            row.update({f"student_{key}": value for key, value in {**score(test, all_test, pred), **timing}.items()})
            row["configuration"] = json.dumps(DEFAULTS[name], sort_keys=True)
        pd.DataFrame(model_rows).to_csv(args.out / "model_comparison.csv", index=False)
        base = next(row for row in model_rows if row["model"] == winner)
        tuned, raw_test, times = fit_predict(winner, best, train, full_train, test, all_test)
        calibrator, raw_oof, calibrated_oof, fold_ids = calibrate(winner, best, train, folds, args.cap)
        calibrated = SigmoidScorer(tuned, calibrator)
        start = time.perf_counter()
        final_test = calibrated.predict_proba(test.x)[:, 1]
        calibration_infer = time.perf_counter() - start
        changes = [{"stage": "default", **{k: v for k, v in base.items() if k.startswith(("cv_", "student_"))}},
                   {"stage": "tuned_raw", "cv_top1": study.best_value, **study.best_trial.user_attrs,
                    **{f"student_{key}": value for key, value in {**score(test, all_test, raw_test), **times}.items()}},
                   {"stage": "tuned_sigmoid", "student_inference_s": calibration_infer,
                    **{f"student_{key}": value for key, value in score(test, all_test, final_test).items()}}]
        changes[1]["cv_macro_f1"] = study.best_trial.user_attrs["cv_macro_f1"]
        pd.DataFrame(changes).to_csv(args.out / "bayesian_comparison.csv", index=False)
        np.savez_compressed(args.scratch / "decoder_scores.npz", train=calibrated_oof,
                            student_test=final_test.astype(np.float32), folds=fold_ids)
        joblib.dump({
            "model": calibrated, "scaler": None, "needs_scaling": False, "feature_cols": features,
            "feature_schema_version": FEATURE_VERSION, "model_version": f"{winner.lower()}-tpe-sigmoid-v2",
            "calibration_method": "sigmoid held-out training trip scores", "sklearn_version": sklearn.__version__,
        }, args.model, compress=3)
        print("Saved", args.model, flush=True)


if __name__ == "__main__":
    main()
