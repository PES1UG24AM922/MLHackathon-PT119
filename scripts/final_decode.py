"""Tune the team's HMM on trip-isolated OOF scores and evaluate student trips."""
import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from api.evaluate import ordered_route_metrics
from api.maps import offline_folium_map
from api.pipeline import display_route
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.decoder import HMMDecoder
from pathsnap.decoder.emissions import classifier_only_predictions, prepare_emissions
from pathsnap.decoder.evaluation import prediction_segment_map
from pathsnap.decoder.graph import RoadGraph
from pathsnap.decoder.schemas import CandidateScore, DecodedPoint, GPSPoint
from src.eda.data_loader import load_trip, resolve_data_root
from src.features.candidate_gen import projected_coordinates
from scripts.final_train import FEATURE_VERSION, SEED

GRID = [
    {"transition_distance_scale_m": 35.0, "emission_weight": 1.0, "gap_reset_seconds": 30.0},
    {"transition_distance_scale_m": 18.0, "emission_weight": 1.15, "gap_reset_seconds": 30.0},
    {"transition_distance_scale_m": 55.0, "emission_weight": 0.85, "gap_reset_seconds": 45.0},
]


def digest(path, text=False):
    payload = path.read_bytes()
    return hashlib.sha256(payload.replace(b"\r\n", b"\n") if text else payload).hexdigest()


def assemble(data_root, partition, trip_id, frame):
    trip = load_trip(partition, trip_id, data_root / partition, load_reference_route=False)
    x, y, node_x, node_y = projected_coordinates(trip)
    track = trip.track[["lon", "lat", "timestamp"]].to_numpy()
    arcs = trip.arcs.to_numpy(dtype=int)
    graph = RoadGraph(np.column_stack((node_x, node_y)), arcs, trip.nodes.to_numpy(dtype=float))
    gps = [GPSPoint(i, ts, x[i], y[i], lon, lat) for i, (lon, lat, ts) in enumerate(track)]
    scores = [CandidateScore(trip_id, idx, arc, p, projection_ratio=ratio)
              for idx, arc, p, ratio in frame[["pt_idx", "cand_arc_id", "p", "proj_ratio"]].itertuples(index=False)]
    labels = frame.drop_duplicates("pt_idx").sort_values("pt_idx")["gt_arc_id"].to_numpy(dtype=int)
    return trip, graph, gps, scores, labels


def decode_task(job):
    data_root, partition, trip_id, frame, configs, map_path = job
    trip, graph, gps, scores, labels = assemble(data_root, partition, trip_id, frame)
    rows = []
    for index, config in configs:
        decoded = HMMDecoder(graph, config).decode(trip_id, gps, scores)
        prediction = np.array([x if x is not None else -1 for x in decoded.decoded_arc_ids], dtype=int)
        row = {"trip_id": trip_id, "fold": int(frame.fold.iloc[0]), "trial": index,
               "n_points": len(labels), "top1": float(np.mean(prediction == labels)),
               "decode_ms": decoded.inference_time_ms}
        if partition == "student_test":
            reference = np.loadtxt(data_root / partition / trip_id / f"{trip_id}.route",
                                   dtype=int, ndmin=1)
            row.update(compare_trip(trip, graph, gps, scores, labels, decoded, reference, config, map_path))
        rows.append(row)
    return rows


def compare_trip(trip, graph, gps, scores, labels, decoded, reference, config, map_path):
    started = time.perf_counter()
    emissions, _ = prepare_emissions(scores, config)
    classifier = classifier_only_predictions(emissions)
    classifier_ms = (time.perf_counter() - started) * 1000
    classification_segments = prediction_segment_map(gps, classifier, config.gap_reset_seconds)
    baseline_points = [
        DecodedPoint(point.gps_point_idx, classifier.get(point.gps_point_idx), None,
                     classification_segments.get(point.gps_point_idx, -1))
        for point in gps
    ]
    output = {"classifier_ms": classifier_ms}
    arcs = trip.arcs.to_numpy(dtype=int)
    times = trip.track.timestamp.to_numpy(dtype=float)
    connected = (np.diff(times) > 0) & (np.diff(times) <= config.gap_reset_seconds)
    for name, points in (("classifier", baseline_points), ("hmm", decoded.decoded_points)):
        pred = np.array([point.decoded_arc_id if point.decoded_arc_id is not None else -1 for point in points])
        valid = connected & (pred[:-1] >= 0) & (pred[1:] >= 0)
        first, second = pred[:-1][valid], pred[1:][valid]
        adjacency = (first == second) | (arcs[first, 1] == arcs[second, 0])
        geometry, segments, _ = display_route(points, graph, config.max_network_search_m)
        output.update({f"{name}_{k}": v for k, v in ordered_route_metrics(reference, segments).items()})
        output.update({
            f"{name}_top1": float(np.mean(pred == labels)),
            f"{name}_point_macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
            f"{name}_coverage": float(np.mean(pred >= 0)),
            f"{name}_continuity": float(adjacency.mean()) if len(adjacency) else math.nan,
        })
        if name == "hmm" and map_path:
            write_map(map_path, trip, graph, gps, scores, geometry, reference)
    return output


def write_map(path, trip, graph, gps, scores, geometry, reference):
    coords = [[point.latitude, point.longitude] for point in gps]
    view = offline_folium_map(coords)
    network = folium.FeatureGroup(name="Candidate road network").add_to(view)
    for arc in sorted({score.candidate_arc_id for score in scores}):
        folium.PolyLine(graph.arc_geometry_latlon(arc), color="#7b8ca0", opacity=0.32, weight=1).add_to(network)
    track_layer = folium.FeatureGroup(name="Raw GPS").add_to(view)
    folium.PolyLine(coords, color="#273954", weight=2).add_to(track_layer)
    ref_layer = folium.FeatureGroup(name="Student-test reference").add_to(view)
    for arc in reference:
        folium.PolyLine(graph.arc_geometry_latlon(int(arc)), color="#d98229", weight=6, opacity=0.65).add_to(ref_layer)
    result_layer = folium.FeatureGroup(name="HMM predicted route").add_to(view)
    for segment in geometry:
        folium.PolyLine([(p["lat"], p["lon"]) for p in segment], color="#007d70", weight=3).add_to(result_layer)
    folium.LayerControl(collapsed=False).add_to(view)
    view.get_root().header.add_child(folium.Element(
        f"<title>Student-test comparison · {trip.trip_id}</title><meta name='referrer' content='no-referrer'>"
    ))
    path.write_text(view.get_root().render(), encoding="utf-8")


def scored_groups(frame, probabilities, fold=None):
    data = frame[["trip_id", "pt_idx", "cand_arc_id", "proj_ratio", "gt_arc_id"]].copy()
    data["p"] = probabilities
    data["fold"] = fold if fold is not None else 0
    return data.groupby("trip_id", sort=True)


def selected_trips(groups):
    per_fold = {}
    for trip_id, group in groups:
        per_fold.setdefault(int(group.fold.iloc[0]), []).append((group.pt_idx.nunique(), trip_id))
    for fold in sorted(per_fold):
        ranked = sorted(per_fold[fold])
        for fraction in (0.20, 0.55):
            yield ranked[round((len(ranked) - 1) * fraction)][1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--out", default="results/final", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data_root = resolve_data_root(args.data).resolve()
    cache = joblib.load(args.cache)
    estimates = np.load(args.scores)
    train = scored_groups(cache["train_df"], estimates["train"], estimates["folds"])
    test = scored_groups(cache["test_df"], estimates["student_test"])
    del cache
    base = DecoderConfig.from_file("configs/hmm.yaml")
    configs = [(i, replace(base, **settings, decoder_version="2.0.0")) for i, settings in enumerate(GRID)]
    train_jobs = [(data_root, "train", tid, train.get_group(tid), configs, None) for tid in selected_trips(train)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for results in pool.map(decode_task, train_jobs):
            rows.extend(results)
            print("Tuned train trip", results[0]["trip_id"], flush=True)
        trials = pd.DataFrame(rows)
        trials.to_csv(args.out / "decoder_tuning_folds.csv", index=False)
        summary = trials.groupby(["trial", "fold"]).top1.mean().groupby("trial").mean()
        winning = int(summary.idxmax())
        pd.DataFrame([{"trial": i, **settings, "mean_cv_top1": summary.loc[i]}
                      for i, settings in enumerate(GRID)]).to_csv(args.out / "decoder_trials.csv", index=False)
        config = configs[winning][1]
        config_path = Path("configs/hmm.final.json")
        config_path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
        test_jobs = [(data_root, "student_test", tid, group, [(winning, config)],
                      args.out / "student_test_map.html" if tid == "00000000" else None)
                     for tid, group in test]
        outcomes = []
        for result in pool.map(decode_task, test_jobs):
            outcomes.extend(result)
            print("Evaluated student trip", result[0]["trip_id"], flush=True)
    outcomes = pd.DataFrame(outcomes)
    outcomes.to_csv(args.out / "classifier_vs_hmm_by_trip.csv", index=False)
    summary = []
    for model in ("classifier", "hmm"):
        summary.append({
            "method": model,
            "point_top1": np.average(outcomes[f"{model}_top1"], weights=outcomes.n_points),
            "point_macro_f1_trip_mean": outcomes[f"{model}_point_macro_f1"].mean(),
            "continuity_trip_mean": outcomes[f"{model}_continuity"].mean(),
            "ordered_route_f1_trip_mean": outcomes[f"{model}_ordered_route_f1"].mean(),
            "coverage_trip_mean": outcomes[f"{model}_coverage"].mean(),
            "inference_ms_trip_mean": outcomes["decode_ms" if model == "hmm" else "classifier_ms"].mean(),
        })
    pd.DataFrame(summary).to_csv(args.out / "classifier_vs_hmm.csv", index=False)
    model_path = Path("artifacts/final_emission.joblib")
    bundle = joblib.load(model_path)
    manifest = {
        "schema_version": "1.0", "feature_version": FEATURE_VERSION, "dataset_version": "PathSnap-student-materials",
        "model_version": bundle["model_version"], "decoder_version": config.decoder_version,
        "random_seed": SEED, "sklearn_version": bundle["sklearn_version"], "model_file": model_path.as_posix(),
        "model_sha256": digest(model_path), "decoder_config": config_path.as_posix(),
        "decoder_sha256": digest(config_path, text=True), "score_kind": "calibrated", "radius_m": 50.0, "max_candidates": 20,
    }
    Path("configs/inference.final.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
