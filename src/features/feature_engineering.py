"""Spatial and kinematic feature extraction and train-only scaler fitting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.candidate_gen import TripCandidates, point_to_single_segments

FEATURE_COLS: list[str] = [
    "perp_dist_m",
    "perp_dist_diff_min_m",
    "perp_dist_rank",
    "n_candidates",
    "proj_ratio",
    "dist_to_from_node_m",
    "dist_to_to_node_m",
    "arc_len_m",
    "heading_cos",
    "heading_diff_deg",
    "gps_speed_mps",
    "dt_prev_s",
    "prev_pt_dist_m",
    "next_pt_dist_m",
]


def _compute_point_kinematics(
    trk_x: np.ndarray,
    trk_y: np.ndarray,
    ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-point unit heading (hx, hy), speed (m/s), and dt_prev (s)."""
    n = len(trk_x)
    dx = np.zeros(n, dtype=float)
    dy = np.zeros(n, dtype=float)
    dt_prev = np.ones(n, dtype=float)
    speed = np.zeros(n, dtype=float)

    if n >= 2:
        # Central differences for interior, one-sided at endpoints
        dx[0] = trk_x[1] - trk_x[0]
        dy[0] = trk_y[1] - trk_y[0]
        dx[-1] = trk_x[-1] - trk_x[-2]
        dy[-1] = trk_y[-1] - trk_y[-2]
        if n >= 3:
            dx[1:-1] = trk_x[2:] - trk_x[:-2]
            dy[1:-1] = trk_y[2:] - trk_y[:-2]

        step_dx = np.diff(trk_x)
        step_dy = np.diff(trk_y)
        step_dt = np.diff(ts)
        step_dt_safe = np.where(step_dt > 1e-3, step_dt, 1.0)
        step_speed = np.hypot(step_dx, step_dy) / step_dt_safe
        speed[1:] = np.clip(step_speed, 0.0, 60.0)
        speed[0] = speed[1]
        dt_prev[1:] = np.clip(step_dt, 0.0, 60.0)
        dt_prev[0] = dt_prev[1]

    norm = np.hypot(dx, dy)
    hx = np.zeros(n, dtype=float)
    hy = np.zeros(n, dtype=float)
    valid = norm >= 0.5
    hx[valid] = dx[valid] / norm[valid]
    hy[valid] = dy[valid] / norm[valid]

    return hx, hy, speed, dt_prev


def extract_features_for_trip(tc: TripCandidates) -> pd.DataFrame:
    """Build candidate-level DataFrame with spatial & kinematic features for one trip."""
    pt_idx = tc.pt_idx
    arc_id = tc.cand_arc_id

    hx_pt, hy_pt, speed_pt, dt_pt = _compute_point_kinematics(tc.trk_x, tc.trk_y, tc.ts)

    # Candidate arc endpoints
    u_cand = tc.u[arc_id]
    v_cand = tc.v[arc_id]
    ax = tc.node_x[u_cand]
    ay = tc.node_y[u_cand]
    bx = tc.node_x[v_cand]
    by = tc.node_y[v_cand]

    px = tc.trk_x[pt_idx]
    py = tc.trk_y[pt_idx]

    dist_to_from = np.hypot(px - ax, py - ay)
    dist_to_to = np.hypot(px - bx, py - by)
    arc_len = tc.arc_len_m[arc_id]

    # Arc unit direction vector
    adx = bx - ax
    ady = by - ay
    anorm = np.where(arc_len > 1e-6, arc_len, 1.0)
    ahx = adx / anorm
    ahy = ady / anorm

    # Heading alignment
    phx = hx_pt[pt_idx]
    phy = hy_pt[pt_idx]
    pt_has_heading = (phx != 0.0) | (phy != 0.0)
    arc_has_heading = arc_len > 1e-6
    valid_heading = pt_has_heading & arc_has_heading

    heading_cos = np.zeros(len(pt_idx), dtype=float)
    raw_cos = phx * ahy * 0.0 + phx * ahx + phy * ahy
    heading_cos[valid_heading] = np.clip(raw_cos[valid_heading], -1.0, 1.0)
    heading_diff_deg = np.where(
        valid_heading, np.degrees(np.arccos(heading_cos)), 90.0
    )

    # Temporal context: distance of prev and next GPS points to the candidate arc
    prev_idx = np.clip(pt_idx - 1, 0, tc.n_points - 1)
    next_idx = np.clip(pt_idx + 1, 0, tc.n_points - 1)
    prev_pt_dist, _ = point_to_single_segments(
        tc.trk_x[prev_idx], tc.trk_y[prev_idx], ax, ay, bx, by
    )
    next_pt_dist, _ = point_to_single_segments(
        tc.trk_x[next_idx], tc.trk_y[next_idx], ax, ay, bx, by
    )

    # Per-point relative features (min distance, rank, count)
    df = pd.DataFrame(
        {
            "split": tc.split,
            "trip_id": tc.trip_id,
            "pt_idx": pt_idx,
            "cand_arc_id": arc_id,
            "gt_arc_id": tc.gt_arc_ids[pt_idx],
            "is_true_arc": tc.is_true_arc,
            "perp_dist_m": tc.perp_dist_m,
            "proj_ratio": tc.proj_ratio,
            "dist_to_from_node_m": dist_to_from,
            "dist_to_to_node_m": dist_to_to,
            "arc_len_m": arc_len,
            "heading_cos": heading_cos,
            "heading_diff_deg": heading_diff_deg,
            "gps_speed_mps": speed_pt[pt_idx],
            "dt_prev_s": dt_pt[pt_idx],
            "prev_pt_dist_m": prev_pt_dist,
            "next_pt_dist_m": next_pt_dist,
        }
    )

    grp = df.groupby("pt_idx")["perp_dist_m"]
    df["perp_dist_diff_min_m"] = df["perp_dist_m"] - grp.transform("min")
    df["perp_dist_rank"] = grp.rank(method="min").astype(float)
    df["n_candidates"] = grp.transform("count").astype(float)

    return df


def fit_and_apply_scaler(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str] = FEATURE_COLS,
    scaler_path: Path | str = "artifacts/feature_scaler.joblib",
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler strictly on train_df and transform both train_df and test_df."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[list(feature_cols)].to_numpy(dtype=np.float32))
    x_test = scaler.transform(test_df[list(feature_cols)].to_numpy(dtype=np.float32))

    out_path = Path(scaler_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_path)
    return x_train, x_test, scaler
