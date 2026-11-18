"""Run label-free HMM inference for one manifest entry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.decoder import DecoderConfig, HMMDecoder  # noqa: E402
from pathsnap.decoder.experiments import load_offline_trip  # noqa: E402
from pathsnap.decoder.io import read_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode one PathSnap trip without labels.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "hmm.yaml"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = read_manifest(manifest_path)
    entry = next((item for item in manifest["trips"] if item["trip_id"] == args.trip_id), None)
    if entry is None:
        raise SystemExit(f"Trip {args.trip_id!r} is not present in manifest.")
    trip = load_offline_trip(entry, manifest_path.parent)
    result = HMMDecoder(trip.graph, DecoderConfig.from_file(args.config)).decode(
        trip.trip_id, trip.gps_points, trip.candidates
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trip_id", "gps_point_idx", "decoded_arc_id", "log_score", "segment_id"],
        )
        writer.writeheader()
        for point in result.decoded_points:
            writer.writerow({"trip_id": result.trip_id, **point.__dict__})
    output.with_suffix(".warnings.json").write_text(
        json.dumps([warning.__dict__ for warning in result.warnings], indent=2), encoding="utf-8"
    )
    print(f"Decoded {len(result.decoded_points)} GPS points in {result.inference_time_ms:.1f} ms: {output}")


if __name__ == "__main__":
    main()
