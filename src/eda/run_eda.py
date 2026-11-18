"""Exploratory Data Analysis (EDA) for PathSnap train and student_test datasets.

Projects coordinates to UTM meters, checks missing values, timestamp gaps,
graph disconnected components, and saves summary tables and Folium maps to `results/`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import folium
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

# Allow running directly as `python src/eda/run_eda.py`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eda.data_loader import TripData, list_trip_ids, load_trip, resolve_data_root


def get_utm_epsg(lon: float, lat: float) -> int:
    """Compute the WGS84 UTM EPSG code for a given (lon, lat)."""
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0.0 else 32700) + zone


def project_to_utm(
    lon: np.ndarray,
    lat: np.ndarray,
    transformer: Transformer,
) -> tuple[np.ndarray, np.ndarray]:
    """Project (lon, lat) arrays in EPSG:4326 to UTM (x, y) in meters."""
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def point_to_segments_min_dist(
    pts_xy: np.ndarray,
    seg_a_xy: np.ndarray,
    seg_b_xy: np.ndarray,
) -> np.ndarray:
    """Compute minimum Euclidean distance (m) from each point to a set of line segments."""
    if len(pts_xy) == 0 or len(seg_a_xy) == 0:
        return np.full(len(pts_xy), np.nan, dtype=float)

    ab = seg_b_xy - seg_a_xy  # (M, 2)
    ab_len2 = np.sum(ab * ab, axis=1)  # (M,)
    ab_len2_safe = np.where(ab_len2 > 1e-12, ab_len2, 1.0)

    min_dists = np.empty(len(pts_xy), dtype=float)
    # Process in chunks to keep memory bounded
    chunk_size = 256
    for start in range(0, len(pts_xy), chunk_size):
        p = pts_xy[start : start + chunk_size, None, :]  # (C, 1, 2)
        ap = p - seg_a_xy[None, :, :]  # (C, M, 2)
        t = np.sum(ap * ab[None, :, :], axis=2) / ab_len2_safe[None, :]  # (C, M)
        t = np.clip(t, 0.0, 1.0)
        proj = seg_a_xy[None, :, :] + t[:, :, None] * ab[None, :, :]  # (C, M, 2)
        d2 = np.sum((p - proj) ** 2, axis=2)
        min_dists[start : start + chunk_size] = np.sqrt(np.min(d2, axis=1))
    return min_dists


def analyze_trip(trip: TripData) -> dict[str, Any]:
    """Run coordinate projection, quality checks, timestamp, and graph EDA on one trip."""
    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    n_track = len(track)
    n_nodes = len(nodes)
    n_arcs = len(arcs)
    n_route = 0 if route is None else len(route)

    # 1. Missing values & bounds checks
    track_na = int(track.isna().sum().sum())
    nodes_na = int(nodes.isna().sum().sum())
    arcs_na = int(arcs.isna().sum().sum())
    nonfinite_track = int((~np.isfinite(track.to_numpy(dtype=float))).sum())
    nonfinite_nodes = int((~np.isfinite(nodes.to_numpy(dtype=float))).sum())

    invalid_arc_nodes = 0
    if n_arcs > 0 and n_nodes > 0:
        u = arcs["from_node"].to_numpy()
        v = arcs["to_node"].to_numpy()
        invalid_arc_nodes = int(
            np.sum((u < 0) | (u >= n_nodes) | (v < 0) | (v >= n_nodes))
        )

    invalid_route_arcs = 0
    if route is not None and len(route) > 0:
        invalid_route_arcs = int(np.sum((route < 0) | (route >= n_arcs)))

    missing_or_invalid_total = (
        track_na
        + nodes_na
        + arcs_na
        + nonfinite_track
        + nonfinite_nodes
        + invalid_arc_nodes
        + invalid_route_arcs
    )

    # 2. UTM projection
    center_lon = float(track["lon"].median())
    center_lat = float(track["lat"].median())
    utm_epsg = get_utm_epsg(center_lon, center_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)

    trk_x, trk_y = project_to_utm(
        track["lon"].to_numpy(), track["lat"].to_numpy(), transformer
    )
    node_x, node_y = project_to_utm(
        nodes["lon"].to_numpy(), nodes["lat"].to_numpy(), transformer
    )

    # Track lengths & speed
    if n_track >= 2:
        dx = np.diff(trk_x)
        dy = np.diff(trk_y)
        step_dist_m = np.hypot(dx, dy)
        track_len_m = float(np.sum(step_dist_m))
        straight_disp_m = float(np.hypot(trk_x[-1] - trk_x[0], trk_y[-1] - trk_y[0]))
    else:
        step_dist_m = np.array([], dtype=float)
        track_len_m = 0.0
        straight_disp_m = 0.0

    # 3. Timestamp gaps
    ts = track["timestamp"].to_numpy(dtype=float)
    if n_track >= 2:
        dt = np.diff(ts)
        duration_s = float(ts[-1] - ts[0])
        dt_min = float(np.min(dt))
        dt_median = float(np.median(dt))
        dt_mean = float(np.mean(dt))
        dt_p95 = float(np.percentile(dt, 95))
        dt_max = float(np.max(dt))
        n_dt_nonpos = int(np.sum(dt <= 0.0))
        n_dt_gap_gt5s = int(np.sum(dt > 5.0))
        n_dt_gap_gt15s = int(np.sum(dt > 15.0))
        n_dt_gap_gt30s = int(np.sum(dt > 30.0))
        valid_dt = dt > 0
        speeds_mps = np.where(valid_dt, step_dist_m / np.where(valid_dt, dt, 1.0), np.nan)
        speed_median_mps = float(np.nanmedian(speeds_mps)) if np.any(valid_dt) else np.nan
        speed_max_mps = float(np.nanmax(speeds_mps)) if np.any(valid_dt) else np.nan
    else:
        duration_s = 0.0
        dt_min = dt_median = dt_mean = dt_p95 = dt_max = np.nan
        n_dt_nonpos = n_dt_gap_gt5s = n_dt_gap_gt15s = n_dt_gap_gt30s = 0
        speed_median_mps = speed_max_mps = np.nan

    # 4. Graph structure & disconnected components
    g = nx.DiGraph()
    g.add_nodes_from(range(n_nodes))
    u_arr = arcs["from_node"].to_numpy()
    v_arr = arcs["to_node"].to_numpy()
    g.add_edges_from(zip(u_arr.tolist(), v_arr.tolist()))

    n_self_loops = int(np.sum(u_arr == v_arr))
    n_parallel_arcs = int(n_arcs - g.number_of_edges())

    wcc = list(nx.weakly_connected_components(g))
    n_wcc = len(wcc)
    largest_wcc_size = max((len(c) for c in wcc), default=0)
    largest_wcc_pct = (100.0 * largest_wcc_size / n_nodes) if n_nodes > 0 else 0.0

    scc = list(nx.strongly_connected_components(g))
    n_scc = len(scc)
    largest_scc_size = max((len(c) for c in scc), default=0)
    largest_scc_pct = (100.0 * largest_scc_size / n_nodes) if n_nodes > 0 else 0.0

    # Arc lengths in UTM meters
    arc_dx = node_x[v_arr] - node_x[u_arr]
    arc_dy = node_y[v_arr] - node_y[u_arr]
    arc_len_m = np.hypot(arc_dx, arc_dy)

    # 5. Ground-truth route metrics
    route_len_m = np.nan
    route_contiguous_pct = np.nan
    mean_gps_to_route_m = np.nan
    p95_gps_to_route_m = np.nan
    max_gps_to_route_m = np.nan

    if route is not None and len(route) > 0 and invalid_route_arcs == 0:
        route_len_m = float(np.sum(arc_len_m[route]))
        if len(route) >= 2:
            r_to = v_arr[route[:-1]]
            r_from = u_arr[route[1:]]
            route_contiguous_pct = float(100.0 * np.mean(r_to == r_from))
        else:
            route_contiguous_pct = 100.0

        pts_xy = np.column_stack([trk_x, trk_y])
        seg_a_xy = np.column_stack([node_x[u_arr[route]], node_y[u_arr[route]]])
        seg_b_xy = np.column_stack([node_x[v_arr[route]], node_y[v_arr[route]]])
        dists = point_to_segments_min_dist(pts_xy, seg_a_xy, seg_b_xy)
        mean_gps_to_route_m = float(np.mean(dists))
        p95_gps_to_route_m = float(np.percentile(dists, 95))
        max_gps_to_route_m = float(np.max(dists))

    return {
        "split": trip.split,
        "trip_id": trip.trip_id,
        "n_track_pts": n_track,
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "n_route_arcs": n_route,
        "utm_epsg": utm_epsg,
        "center_lon": round(center_lon, 6),
        "center_lat": round(center_lat, 6),
        "missing_or_invalid": missing_or_invalid_total,
        "duration_s": round(duration_s, 2),
        "dt_min": round(dt_min, 3),
        "dt_median": round(dt_median, 3),
        "dt_mean": round(dt_mean, 3),
        "dt_p95": round(dt_p95, 3),
        "dt_max": round(dt_max, 3),
        "n_dt_nonpos": n_dt_nonpos,
        "n_dt_gap_gt5s": n_dt_gap_gt5s,
        "n_dt_gap_gt15s": n_dt_gap_gt15s,
        "n_dt_gap_gt30s": n_dt_gap_gt30s,
        "track_len_m": round(track_len_m, 2),
        "straight_disp_m": round(straight_disp_m, 2),
        "route_len_m": round(route_len_m, 2) if np.isfinite(route_len_m) else np.nan,
        "speed_median_mps": round(speed_median_mps, 2) if np.isfinite(speed_median_mps) else np.nan,
        "speed_max_mps": round(speed_max_mps, 2) if np.isfinite(speed_max_mps) else np.nan,
        "n_self_loops": n_self_loops,
        "n_parallel_arcs": n_parallel_arcs,
        "n_wcc": n_wcc,
        "largest_wcc_pct": round(largest_wcc_pct, 2),
        "n_scc": n_scc,
        "largest_scc_pct": round(largest_scc_pct, 2),
        "route_contiguous_pct": (
            round(route_contiguous_pct, 2) if np.isfinite(route_contiguous_pct) else np.nan
        ),
        "mean_gps_to_route_m": (
            round(mean_gps_to_route_m, 2) if np.isfinite(mean_gps_to_route_m) else np.nan
        ),
        "p95_gps_to_route_m": (
            round(p95_gps_to_route_m, 2) if np.isfinite(p95_gps_to_route_m) else np.nan
        ),
        "max_gps_to_route_m": (
            round(max_gps_to_route_m, 2) if np.isfinite(max_gps_to_route_m) else np.nan
        ),
    }


def build_overview_map(
    trips: list[TripData],
    summary_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Create a Folium overview map of all train and student_test trips."""
    center_lat = float(summary_df["center_lat"].median())
    center_lon = float(summary_df["center_lon"].median())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    train_group = folium.FeatureGroup(name="Train Trips (80)", show=True)
    test_group = folium.FeatureGroup(name="Student Test Trips (10)", show=True)

    meta_lookup = {
        (row["split"], row["trip_id"]): row
        for _, row in summary_df.iterrows()
    }

    for trip in trips:
        row = meta_lookup[(trip.split, trip.trip_id)]
        coords = list(zip(trip.track["lat"].tolist(), trip.track["lon"].tolist()))
        is_test = trip.split == "student_test"
        color = "#e6550d" if is_test else "#3182bd"
        group = test_group if is_test else train_group

        popup_html = (
            f"<b>{trip.split} / {trip.trip_id}</b><br>"
            f"Track pts: {row['n_track_pts']} | Duration: {row['duration_s']} s<br>"
            f"Track len: {row['track_len_m']} m | Route len: {row['route_len_m']} m<br>"
            f"WCC: {row['n_wcc']} ({row['largest_wcc_pct']}% in largest)<br>"
            f"Max dt gap: {row['dt_max']} s | Mean GPS err: {row['mean_gps_to_route_m']} m"
        )
        folium.PolyLine(
            locations=coords,
            color=color,
            weight=3 if is_test else 2,
            opacity=0.85 if is_test else 0.45,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{trip.split}/{trip.trip_id}",
        ).add_to(group)

    train_group.add_to(m)
    test_group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))


def build_trip_detail_map(
    trip: TripData,
    row: pd.Series,
    out_path: Path,
    pad_deg: float = 0.004,
) -> None:
    """Create a detailed Folium map for one trip: local road graph, ground-truth route, and GPS track."""
    trk_lat = trip.track["lat"].to_numpy()
    trk_lon = trip.track["lon"].to_numpy()
    center_lat = float(np.median(trk_lat))
    center_lon = float(np.median(trk_lon))

    lat_min, lat_max = float(trk_lat.min()) - pad_deg, float(trk_lat.max()) + pad_deg
    lon_min, lon_max = float(trk_lon.min()) - pad_deg, float(trk_lon.max()) + pad_deg

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Local road network arcs inside bounding box
    node_lon = trip.nodes["lon"].to_numpy()
    node_lat = trip.nodes["lat"].to_numpy()
    u = trip.arcs["from_node"].to_numpy()
    v = trip.arcs["to_node"].to_numpy()

    in_bbox = (
        (node_lat[u] >= lat_min)
        & (node_lat[u] <= lat_max)
        & (node_lon[u] >= lon_min)
        & (node_lon[u] <= lon_max)
        & (node_lat[v] >= lat_min)
        & (node_lat[v] <= lat_max)
        & (node_lon[v] >= lon_min)
        & (node_lon[v] <= lon_max)
    )
    local_indices = np.where(in_bbox)[0]
    if len(local_indices) > 4000:
        local_indices = local_indices[:4000]

    road_group = folium.FeatureGroup(name=f"Local Road Graph ({len(local_indices)} arcs)", show=True)
    for idx in local_indices:
        ui, vi = int(u[idx]), int(v[idx])
        folium.PolyLine(
            locations=[(float(node_lat[ui]), float(node_lon[ui])), (float(node_lat[vi]), float(node_lon[vi]))],
            color="#969696",
            weight=1.5,
            opacity=0.5,
        ).add_to(road_group)
    road_group.add_to(m)

    # Ground-truth route
    if trip.route is not None and len(trip.route) > 0:
        route_group = folium.FeatureGroup(name=f"Ground-Truth Route ({len(trip.route)} arcs)", show=True)
        for arc_id in trip.route:
            ui, vi = int(u[arc_id]), int(v[arc_id])
            folium.PolyLine(
                locations=[(float(node_lat[ui]), float(node_lon[ui])), (float(node_lat[vi]), float(node_lon[vi]))],
                color="#2ca25f",
                weight=5,
                opacity=0.85,
                tooltip=f"arc_id={arc_id}",
            ).add_to(route_group)
        route_group.add_to(m)

    # Raw GPS track
    gps_group = folium.FeatureGroup(name=f"GPS Track ({len(trk_lat)} pts)", show=True)
    coords = list(zip(trk_lat.tolist(), trk_lon.tolist()))
    folium.PolyLine(locations=coords, color="#de2d26", weight=2.5, opacity=0.8).add_to(gps_group)

    ts = trip.track["timestamp"].to_numpy()
    for i, (lat, lon, t) in enumerate(zip(trk_lat, trk_lon, ts)):
        is_gap = i > 0 and (t - ts[i - 1]) > 5.0
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=5 if is_gap else 2.5,
            color="#a50f15" if is_gap else "#de2d26",
            fill=True,
            fill_opacity=0.9,
            tooltip=f"pt {i} | t={t:.1f}s" + (f" (gap={t - ts[i - 1]:.1f}s)" if is_gap else ""),
        ).add_to(gps_group)

    folium.Marker(
        location=[float(trk_lat[0]), float(trk_lon[0])],
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(gps_group)
    folium.Marker(
        location=[float(trk_lat[-1]), float(trk_lon[-1])],
        tooltip="End",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(gps_group)
    gps_group.add_to(m)

    title_html = (
        f'<div style="position: fixed; top: 10px; left: 50px; z-index: 9999; '
        f'background: white; padding: 8px 12px; border-radius: 6px; '
        f'box-shadow: 0 2px 6px rgba(0,0,0,0.25); font-family: sans-serif; font-size: 12px;">'
        f'<b>{trip.split} / {trip.trip_id}</b> | '
        f'Pts: {row["n_track_pts"]} | Dur: {row["duration_s"]}s | '
        f'Max dt: {row["dt_max"]}s | Mean GPS err: {row["mean_gps_to_route_m"]}m'
        f"</div>"
    )
    m.get_root().html.add_child(folium.Element(title_html))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))


def run_eda(base_dir: Path | str = ".", results_dir: Path | str = "results") -> pd.DataFrame:
    """Execute full EDA across train and student_test splits and save outputs."""
    data_root = resolve_data_root(base_dir)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trips: list[TripData] = []
    records: list[dict[str, Any]] = []

    for split in ("train", "student_test"):
        split_dir = data_root / split
        trip_ids = list_trip_ids(split_dir)
        for tid in trip_ids:
            trip = load_trip(split, tid, split_dir)
            trips.append(trip)
            records.append(analyze_trip(trip))

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(out_dir / "eda_trip_summary.csv", index=False)

    # Split-level aggregate table
    agg_cols = [
        "n_track_pts",
        "n_nodes",
        "n_arcs",
        "n_route_arcs",
        "missing_or_invalid",
        "duration_s",
        "dt_median",
        "dt_max",
        "n_dt_nonpos",
        "n_dt_gap_gt5s",
        "n_dt_gap_gt15s",
        "track_len_m",
        "route_len_m",
        "n_wcc",
        "largest_wcc_pct",
        "n_scc",
        "route_contiguous_pct",
        "mean_gps_to_route_m",
        "max_gps_to_route_m",
    ]
    split_summary = (
        summary_df.groupby("split")[agg_cols]
        .agg(["count", "mean", "median", "min", "max"])
        .round(3)
    )
    split_summary.columns = [f"{col}_{stat}" for col, stat in split_summary.columns]
    split_summary.to_csv(out_dir / "eda_split_summary.csv")

    # Anomalies table
    anomaly_mask = (
        (summary_df["missing_or_invalid"] > 0)
        | (summary_df["n_dt_nonpos"] > 0)
        | (summary_df["n_dt_gap_gt15s"] > 0)
        | (summary_df["route_contiguous_pct"] < 100.0)
        | (summary_df["mean_gps_to_route_m"] > 25.0)
    )
    anomalies_df = summary_df[anomaly_mask].copy()
    anomalies_df.to_csv(out_dir / "eda_anomalies.csv", index=False)

    # Folium maps
    build_overview_map(trips, summary_df, out_dir / "eda_overview_map.html")

    trip_lookup = {(t.split, t.trip_id): t for t in trips}

    # Sample train trip (median track length)
    train_df = summary_df[summary_df["split"] == "train"].sort_values("n_track_pts")
    sample_train_row = train_df.iloc[len(train_df) // 2]
    build_trip_detail_map(
        trip_lookup[("train", sample_train_row["trip_id"])],
        sample_train_row,
        out_dir / f"eda_sample_train_{sample_train_row['trip_id']}.html",
    )

    # Sample student_test trip (median track length)
    test_df = summary_df[summary_df["split"] == "student_test"].sort_values("n_track_pts")
    sample_test_row = test_df.iloc[len(test_df) // 2]
    build_trip_detail_map(
        trip_lookup[("student_test", sample_test_row["trip_id"])],
        sample_test_row,
        out_dir / f"eda_sample_test_{sample_test_row['trip_id']}.html",
    )

    # Notable trip with largest timestamp gap
    max_gap_row = summary_df.sort_values("dt_max", ascending=False).iloc[0]
    build_trip_detail_map(
        trip_lookup[(max_gap_row["split"], max_gap_row["trip_id"])],
        max_gap_row,
        out_dir / f"eda_max_gap_{max_gap_row['split']}_{max_gap_row['trip_id']}.html",
    )

    # Print concise report to stdout
    print(f"Data root: {data_root.resolve()}")
    print(f"Total trips analyzed: {len(summary_df)}")
    print(summary_df.groupby("split")[["n_track_pts", "n_nodes", "n_arcs", "n_route_arcs", "missing_or_invalid", "n_dt_nonpos", "n_dt_gap_gt5s", "n_dt_gap_gt15s", "n_wcc", "largest_wcc_pct", "route_contiguous_pct", "mean_gps_to_route_m"]].agg(["count", "mean", "min", "max"]).to_string())
    print(f"\nAnomalous trips flagged: {len(anomalies_df)}")
    print(f"Saved outputs to: {out_dir.resolve()}")
    return summary_df


if __name__ == "__main__":
    run_eda()
