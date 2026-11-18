"""Offline Folium map generation from actual classifier and HMM results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.decoder import DecoderConfig, HMMDecoder  # noqa: E402
from pathsnap.decoder.emissions import classifier_only_predictions, prepare_emissions  # noqa: E402
from pathsnap.decoder.experiments import load_offline_trip  # noqa: E402
from pathsnap.decoder.io import read_manifest  # noqa: E402
from pathsnap.decoder.visualization import build_folium_map  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an offline PathSnap Folium comparison map.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "hmm.yaml"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-reference", action="store_true", help="Offline report use only")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = read_manifest(manifest_path)
    entry = next((item for item in manifest["trips"] if item["trip_id"] == args.trip_id), None)
    if entry is None:
        raise SystemExit(f"Trip {args.trip_id!r} is not present in manifest.")
    trip = load_offline_trip(entry, manifest_path.parent, require_labels=args.include_reference)
    config = DecoderConfig.from_file(args.config)
    emissions, _ = prepare_emissions(trip.candidates, config)
    result = HMMDecoder(trip.graph, config).decode(trip.trip_id, trip.gps_points, trip.candidates)
    reference = None
    if args.include_reference and trip.reference_labels:
        reference = [trip.reference_labels[index] for index in sorted(trip.reference_labels)]
    route_map = build_folium_map(
        trip.graph,
        trip.gps_points,
        classifier_only_predictions(emissions),
        result,
        reference,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    route_map.save(output)
    print(f"Saved map to {output}")


if __name__ == "__main__":
    main()
