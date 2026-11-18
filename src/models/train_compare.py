"""Train and compare >=5 model families using GroupKFold (grouped by trip) and evaluate on student_test."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.build_dataset import build_feature_datasets
from src.features.feature_engineering import FEATURE_COLS


@dataclass
class ModelSpec:
    name: str
    family: str
    estimator_fn: Callable[[], object]
    needs_scaling: bool
    max_train_rows: int | None = None
    notes: str = ""


def compute_top1_accuracy(df: pd.DataFrame, probs: np.ndarray) -> float:
    """Compute per-GPS-point Top-1 candidate arc accuracy."""
    eval_df = pd.DataFrame(
        {
            "trip_id": df["trip_id"].to_numpy(),
            "pt_idx": df["pt_idx"].to_numpy(),
            "is_true_arc": df["is_true_arc"].to_numpy(),
            "prob": probs,
        }
    )
    # Pick candidate row with maximum predicted probability for each (trip_id, pt_idx)
    best_idx = eval_df.groupby(["trip_id", "pt_idx"], sort=False)["prob"].idxmax()
    return float(eval_df.loc[best_idx, "is_true_arc"].mean())


def evaluate_predictions(
    df: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
) -> dict[str, float]:
    """Compute Top-1 Accuracy, Macro F1, and Log Loss."""
    probs_clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    y_pred = (probs_clipped >= 0.5).astype(int)
    return {
        "top1_acc": compute_top1_accuracy(df, probs_clipped),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_true, probs_clipped, labels=[0, 1])),
    }


def _subsample_stratified(
    x: np.ndarray,
    y: np.ndarray,
    max_rows: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically subsample up to `max_rows` preserving class balance."""
    if len(y) <= max_rows:
        return x, y
    rng = np.random.default_rng(seed)
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    frac = max_rows / len(y)
    n_pos = max(1, int(round(len(idx_pos) * frac)))
    n_neg = max(1, max_rows - n_pos)
    chosen_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
    chosen_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
    chosen = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(chosen)
    return x[chosen], y[chosen]


def get_model_specs() -> list[ModelSpec]:
    """Define at least 5 distinct model families for comparison."""
    return [
        ModelSpec(
            name="LogisticRegression",
            family="Linear",
            estimator_fn=lambda: LogisticRegression(max_iter=500, C=1.0, random_state=42),
            needs_scaling=True,
            notes="L2-regularized linear baseline",
        ),
        ModelSpec(
            name="KNeighborsClassifier",
            family="Instance-Based",
            estimator_fn=lambda: KNeighborsClassifier(
                n_neighbors=15, weights="distance", algorithm="kd_tree", n_jobs=-1
            ),
            needs_scaling=True,
            max_train_rows=30_000,
            notes="kd_tree k=15 distance-weighted (30k stratified train subsample)",
        ),
        ModelSpec(
            name="RandomForestClassifier",
            family="Bagging Ensemble",
            estimator_fn=lambda: RandomForestClassifier(
                n_estimators=100,
                max_depth=14,
                min_samples_leaf=5,
                max_samples=200_000,
                n_jobs=-1,
                random_state=42,
            ),
            needs_scaling=False,
            notes="100 trees, max_depth=14, max_samples=200k",
        ),
        ModelSpec(
            name="ExtraTreesClassifier",
            family="Randomized Ensemble",
            estimator_fn=lambda: ExtraTreesClassifier(
                n_estimators=100,
                max_depth=16,
                min_samples_leaf=5,
                bootstrap=True,
                max_samples=200_000,
                n_jobs=-1,
                random_state=42,
            ),
            needs_scaling=False,
            notes="100 extremely randomized trees, max_depth=16, max_samples=200k",
        ),
        ModelSpec(
            name="HistGradientBoosting",
            family="Gradient Boosting",
            estimator_fn=lambda: HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=8,
                learning_rate=0.08,
                random_state=42,
            ),
            needs_scaling=False,
            notes="Histogram-based GBDT (LightGBM-style)",
        ),
        ModelSpec(
            name="MLPClassifier",
            family="Neural Network",
            estimator_fn=lambda: MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=30,
                batch_size=1024,
                early_stopping=True,
                n_iter_no_change=5,
                random_state=42,
            ),
            needs_scaling=True,
            notes="2-layer MLP (64, 32) with early stopping",
        ),
    ]


def run_training_and_comparison(
    artifacts_dir: Path | str = "artifacts",
    results_dir: Path | str = "results",
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run GroupKFold CV on train and evaluate all models on student_test."""
    art_dir = Path(artifacts_dir)
    res_dir = Path(results_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = art_dir / "feature_datasets.joblib"
    if not dataset_path.exists():
        build_feature_datasets(artifacts_dir=art_dir, results_dir=res_dir)

    bundle = joblib.load(dataset_path)
    train_df: pd.DataFrame = bundle["train_df"]
    test_df: pd.DataFrame = bundle["test_df"]
    feature_cols: list[str] = bundle["feature_cols"]

    x_train_raw = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["is_true_arc"].to_numpy(dtype=int)
    groups = train_df["trip_id"].to_numpy()

    x_test_raw = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["is_true_arc"].to_numpy(dtype=int)

    # Pre-fitted scaler on all 80 train trips (for final test evaluation)
    full_scaler: StandardScaler = joblib.load(art_dir / "feature_scaler.joblib")
    x_train_full_scaled = bundle["x_train_scaled"]
    x_test_full_scaled = bundle["x_test_scaled"]

    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(x_train_raw, y_train, groups=groups))

    specs = get_model_specs()
    results_rows: list[dict] = []
    best_model_name = ""
    best_cv_top1 = -1.0
    best_model_obj = None

    for spec in specs:
        print(f"\n=== Training {spec.name} ({spec.family}) ===")
        fold_metrics: list[dict[str, float]] = []
        fold_train_times: list[float] = []
        fold_infer_times: list[float] = []

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            x_tr, y_tr = x_train_raw[tr_idx], y_train[tr_idx]
            x_va, y_va = x_train_raw[va_idx], y_train[va_idx]
            va_df = train_df.iloc[va_idx]

            if spec.needs_scaling:
                fold_scaler = StandardScaler()
                x_tr_in = fold_scaler.fit_transform(x_tr)
                x_va_in = fold_scaler.transform(x_va)
            else:
                x_tr_in, x_va_in = x_tr, x_va

            if spec.max_train_rows is not None:
                x_tr_in, y_tr_in = _subsample_stratified(
                    x_tr_in, y_tr, spec.max_train_rows, seed=42 + fold_idx
                )
            else:
                y_tr_in = y_tr

            model = spec.estimator_fn()
            t0 = time.perf_counter()
            model.fit(x_tr_in, y_tr_in)
            fold_train_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            va_probs = model.predict_proba(x_va_in)[:, 1]
            fold_infer_times.append(time.perf_counter() - t1)

            m = evaluate_predictions(va_df, y_va, va_probs)
            fold_metrics.append(m)

        cv_top1_mean = float(np.mean([m["top1_acc"] for m in fold_metrics]))
        cv_top1_std = float(np.std([m["top1_acc"] for m in fold_metrics]))
        cv_f1_mean = float(np.mean([m["macro_f1"] for m in fold_metrics]))
        cv_f1_std = float(np.std([m["macro_f1"] for m in fold_metrics]))
        cv_ll_mean = float(np.mean([m["log_loss"] for m in fold_metrics]))
        cv_ll_std = float(np.std([m["log_loss"] for m in fold_metrics]))

        # Fit on all 80 train trips and evaluate on student_test (10 trips)
        x_tr_final = x_train_full_scaled if spec.needs_scaling else x_train_raw
        x_te_final = x_test_full_scaled if spec.needs_scaling else x_test_raw

        if spec.max_train_rows is not None:
            x_tr_final_in, y_tr_final_in = _subsample_stratified(
                x_tr_final, y_train, spec.max_train_rows, seed=42
            )
        else:
            x_tr_final_in, y_tr_final_in = x_tr_final, y_train

        final_model = spec.estimator_fn()
        t_fit0 = time.perf_counter()
        final_model.fit(x_tr_final_in, y_tr_final_in)
        full_train_time_s = time.perf_counter() - t_fit0

        t_inf0 = time.perf_counter()
        test_probs = final_model.predict_proba(x_te_final)[:, 1]
        test_infer_time_s = time.perf_counter() - t_inf0

        test_m = evaluate_predictions(test_df, y_test, test_probs)

        row = {
            "model": spec.name,
            "family": spec.family,
            "needs_scaling": spec.needs_scaling,
            "cv_folds": n_splits,
            "cv_top1_acc_mean": round(cv_top1_mean, 5),
            "cv_top1_acc_std": round(cv_top1_std, 5),
            "cv_macro_f1_mean": round(cv_f1_mean, 5),
            "cv_macro_f1_std": round(cv_f1_std, 5),
            "cv_log_loss_mean": round(cv_ll_mean, 5),
            "cv_log_loss_std": round(cv_ll_std, 5),
            "cv_mean_infer_s": round(float(np.mean(fold_infer_times)), 4),
            "test_top1_acc": round(test_m["top1_acc"], 5),
            "test_macro_f1": round(test_m["macro_f1"], 5),
            "test_log_loss": round(test_m["log_loss"], 5),
            "full_train_time_s": round(full_train_time_s, 3),
            "test_infer_time_s": round(test_infer_time_s, 4),
            "notes": spec.notes,
        }
        results_rows.append(row)
        print(
            f"  CV Top-1: {cv_top1_mean:.4f}±{cv_top1_std:.4f} | CV F1: {cv_f1_mean:.4f} | CV LogLoss: {cv_ll_mean:.4f}\n"
            f"  Test Top-1: {test_m['top1_acc']:.4f} | Test F1: {test_m['macro_f1']:.4f} | Test LogLoss: {test_m['log_loss']:.4f} | Infer: {test_infer_time_s:.3f}s"
        )

        if cv_top1_mean > best_cv_top1:
            best_cv_top1 = cv_top1_mean
            best_model_name = spec.name
            best_model_obj = {
                "model_name": spec.name,
                "family": spec.family,
                "needs_scaling": spec.needs_scaling,
                "model": clone(final_model) if False else final_model,
                "scaler": full_scaler if spec.needs_scaling else None,
                "feature_cols": feature_cols,
            }

    results_df = pd.DataFrame(results_rows).sort_values("test_top1_acc", ascending=False)
    results_df.to_csv(res_dir / "model_comparison.csv", index=False)

    if best_model_obj is not None:
        joblib.dump(best_model_obj, art_dir / "best_emission_model.joblib", compress=3)
        print(f"\nSaved best emission model ({best_model_name}) to {art_dir / 'best_emission_model.joblib'}")

    print(f"Saved model comparison table to {res_dir / 'model_comparison.csv'}")
    return results_df


if __name__ == "__main__":
    df = run_training_and_comparison()
    print("\n=== Final Model Comparison ===")
    print(
        df[
            [
                "model",
                "family",
                "cv_top1_acc_mean",
                "cv_macro_f1_mean",
                "cv_log_loss_mean",
                "test_top1_acc",
                "test_macro_f1",
                "test_log_loss",
                "test_infer_time_s",
            ]
        ].to_string(index=False)
    )
