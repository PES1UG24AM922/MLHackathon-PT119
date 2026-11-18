"""Candidate arc generation within a UTM search radius and Miss Rate logging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

from src.eda.data_loader import TripData
from src.eda.coordinates import get_utm_epsg, project_to_utm


@dataclass
class TripCandidates:
    """Candidate arcs and ground-truth alignment for a single trip."""

    split: str
    trip_id: str
    n_points: int
    trk_x: np.ndarray
    trk_y: np.ndarray
    ts: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    arc_len_m: np.ndarray
    gt_arc_ids: np.ndarray | None
    # Flattened candidate arrays
    pt_idx: np.ndarray
    cand_arc_id: np.ndarray
    perp_dist_m: np.ndarray
    proj_ratio: np.ndarray
    is_true_arc: np.ndarray | None
    # Summary stats
    n_missed_gt: int | None
    n_zero_cands: int
    miss_rate: float | None
    zero_cand_rate: float


def point_to_single_segments(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized point-to-segment distance and projection ratio t in [0, 1]."""
    abx = bx - ax
    aby = by - ay
    ab_len2 = abx * abx + aby * aby
    safe_len2 = np.where(ab_len2 > 1e-12, ab_len2, 1.0)
    t = ((px - ax) * abx + (py - ay) * aby) / safe_len2
    t = np.clip(t, 0.0, 1.0)
    proj_x = ax + t * abx
    proj_y = ay + t * aby
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t


def assign_ground_truth_arcs(
    trk_x: np.ndarray,
    trk_y: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    route: np.ndarray,
) -> np.ndarray:
    """Assign each GPS point to its nearest ground-truth arc in `route`."""
    ax = node_x[u[route]]
    ay = node_y[u[route]]
    bx = node_x[v[route]]
    by = node_y[v[route]]

    abx = bx - ax
    aby = by - ay
    ab_len2 = abx * abx + aby * aby
    safe_len2 = np.where(ab_len2 > 1e-12, ab_len2, 1.0)

    n_pts = len(trk_x)
    gt_arcs = np.empty(n_pts, dtype=int)
    chunk_size = 256
    for start in range(0, n_pts, chunk_size):
        px = trk_x[start : start + chunk_size, None]
        py = trk_y[start : start + chunk_size, None]
        t = ((px - ax[None, :]) * abx[None, :] + (py - ay[None, :]) * aby[None, :]) / safe_len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        dx = px - (ax[None, :] + t * abx[None, :])
        dy = py - (ay[None, :] + t * aby[None, :])
        d2 = dx * dx + dy * dy
        best_route_idx = np.argmin(d2, axis=1)
        gt_arcs[start : start + chunk_size] = route[best_route_idx]
    return gt_arcs


def _build_arc_sample_tree(
    node_x: np.ndarray,
    node_y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    step_m: float = 20.0,
) -> Tuple[cKDTree, np.ndarray]:
    """Subdivide arcs every `step_m` meters and build a cKDTree mapping samples to arc_id."""
    ax = node_x[u]
    ay = node_y[u]
    bx = node_x[v]
    by = node_y[v]
    lengths = np.hypot(bx - ax, by - ay)

    # Number of sample points per arc (at least 2: start and end; plus midpoint if short)
    n_samples = np.maximum(2, np.ceil(lengths / step_m).astype(int) + 1)
    # Cap extreme outlier arcs to avoid memory spikes
    n_samples = np.minimum(n_samples, 200)

    arc_ids_rep = np.repeat(np.arange(len(u), dtype=int), n_samples)

    # Vectorized parameter t for each repeated arc
    # Compute cumulative offsets to build t in [0, 1]
    total = int(np.sum(n_samples))
    t_vals = np.empty(total, dtype=float)
    idx = 0
    for k in n_samples:
        t_vals[idx : idx + k] = np.linspace(0.0, 1.0, k)
        idx += k

    sx = ax[arc_ids_rep] + t_vals * (bx[arc_ids_rep] - ax[arc_ids_rep])
    sy = ay[arc_ids_rep] + t_vals * (by[arc_ids_rep] - ay[arc_ids_rep])
    tree = cKDTree(np.column_stack([sx, sy]))
    return tree, arc_ids_rep


def generate_candidates_for_trip(
    trip: TripData,
    radius_m: float = 50.0,
    max_candidates: int = 20,
    sample_step_m: float = 20.0,
    require_ground_truth: bool = True,
) -> TripCandidates:
    """Project a trip and retrieve candidates, with optional supervised labels.

    Set ``require_ground_truth=False`` in the final prediction/API path.  That
    route never reads `.route`, so evaluator labels cannot enter inference.
    """
    center_lon = float(trip.track["lon"].median())
    center_lat = float(trip.track["lat"].median())
    epsg = get_utm_epsg(center_lon, center_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    trk_x, trk_y = project_to_utm(
        trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy(), transformer
    )
    ts = trip.track["timestamp"].to_numpy(dtype=float)
    node_x, node_y = project_to_utm(
        trip.nodes["lon"].to_numpy(), trip.nodes["lat"].to_numpy(), transformer
    )

    u = trip.arcs["from_node"].to_numpy(dtype=int)
    v = trip.arcs["to_node"].to_numpy(dtype=int)
    arc_len_m = np.hypot(node_x[v] - node_x[u], node_y[v] - node_y[u])

    if require_ground_truth and (trip.route is None or len(trip.route) == 0):
        raise ValueError(f"Trip {trip.split}/{trip.trip_id} has no ground-truth route.")
    gt_arc_ids = (
        assign_ground_truth_arcs(trk_x, trk_y, node_x, node_y, u, v, trip.route)
        if require_ground_truth and trip.route is not None
        else None
    )

    tree, sample_arc_ids = _build_arc_sample_tree(node_x, node_y, u, v, step_m=sample_step_m)
    query_radius = radius_m + 0.6 * sample_step_m
    pts_xy = np.column_stack([trk_x, trk_y])
    neighbors_list = tree.query_ball_point(pts_xy, r=query_radius)

    ax = node_x[u]
    ay = node_y[u]
    bx = node_x[v]
    by = node_y[v]

    pt_indices: list[np.ndarray] = []
    cand_arcs: list[np.ndarray] = []
    perp_dists: list[np.ndarray] = []
    proj_ratios: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    n_missed_gt = 0
    n_zero_cands = 0

    for i, nbrs in enumerate(neighbors_list):
        if len(nbrs) > 0:
            unique_arcs = np.unique(sample_arc_ids[nbrs])
            px = np.full(len(unique_arcs), trk_x[i], dtype=float)
            py = np.full(len(unique_arcs), trk_y[i], dtype=float)
            dists, ratios = point_to_single_segments(
                px, py, ax[unique_arcs], ay[unique_arcs], bx[unique_arcs], by[unique_arcs]
            )
            within = dists <= radius_m
            unique_arcs = unique_arcs[within]
            dists = dists[within]
            ratios = ratios[within]
        else:
            unique_arcs = np.empty(0, dtype=int)
            dists = np.empty(0, dtype=float)
            ratios = np.empty(0, dtype=float)

        gt_arc = int(gt_arc_ids[i]) if gt_arc_ids is not None else None
        if gt_arc is not None and gt_arc not in unique_arcs:
            n_missed_gt += 1

        if len(unique_arcs) == 0:
            n_zero_cands += 1
            # Fallback to single nearest sample arc so every GPS point has >= 1 candidate
            _, nn_idx = tree.query(pts_xy[i], k=1)
            fallback_arc = int(sample_arc_ids[nn_idx])
            d_fb, r_fb = point_to_single_segments(
                np.array([trk_x[i]]),
                np.array([trk_y[i]]),
                np.array([ax[fallback_arc]]),
                np.array([ay[fallback_arc]]),
                np.array([bx[fallback_arc]]),
                np.array([by[fallback_arc]]),
            )
            unique_arcs = np.array([fallback_arc], dtype=int)
            dists = d_fb
            ratios = r_fb
        elif len(unique_arcs) > max_candidates:
            order = np.argsort(dists)[:max_candidates]
            unique_arcs = unique_arcs[order]
            dists = dists[order]
            ratios = ratios[order]
            if gt_arc is not None and gt_arc not in unique_arcs:
                # Note: if gt_arc was dropped by max_candidates cap, count as missed if not already
                pass

        # Re-check miss after max_candidates truncation so miss_rate reflects final candidate set
        pt_indices.append(np.full(len(unique_arcs), i, dtype=int))
        cand_arcs.append(unique_arcs)
        perp_dists.append(dists)
        proj_ratios.append(ratios)
        if gt_arc is not None:
            labels.append((unique_arcs == gt_arc).astype(int))

    pt_idx_arr = np.concatenate(pt_indices)
    cand_arc_arr = np.concatenate(cand_arcs)
    perp_dist_arr = np.concatenate(perp_dists)
    proj_ratio_arr = np.concatenate(proj_ratios)
    is_true_arr = np.concatenate(labels) if labels else None

    n_points = len(trk_x)
    # Exact final miss count: points where no candidate has is_true_arc == 1
    # (which equals n_missed_gt unless truncated by max_candidates)
    n_hit = int(np.sum(is_true_arr)) if is_true_arr is not None else 0
    n_missed_final = n_points - n_hit if is_true_arr is not None else None

    return TripCandidates(
        split=trip.split,
        trip_id=trip.trip_id,
        n_points=n_points,
        trk_x=trk_x,
        trk_y=trk_y,
        ts=ts,
        node_x=node_x,
        node_y=node_y,
        u=u,
        v=v,
        arc_len_m=arc_len_m,
        gt_arc_ids=gt_arc_ids,
        pt_idx=pt_idx_arr,
        cand_arc_id=cand_arc_arr,
        perp_dist_m=perp_dist_arr,
        proj_ratio=proj_ratio_arr,
        is_true_arc=is_true_arr,
        n_missed_gt=n_missed_final,
        n_zero_cands=n_zero_cands,
        miss_rate=float(n_missed_final / n_points) if n_missed_final is not None else None,
        zero_cand_rate=float(n_zero_cands / n_points),
    )
