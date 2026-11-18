"""Create a local evaluator config during the official demo."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

INPUT = re.compile(r"(?:(.*)/)?([0-9]{8})/\2\.(track|nodes|arcs|route)")
REQUIRED = {"track", "nodes", "arcs", "route"}


def evaluator_config(dataset, selected_group=None):
    groups = defaultdict(lambda: defaultdict(set))
    if dataset.is_dir():
        names = [path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.is_file()]
    else:
        with ZipFile(dataset) as archive:
            names = archive.namelist()
    for name in names:
        match = INPUT.fullmatch(name)
        if match:
            prefix, trip_id, suffix = match.groups()
            group = prefix or ""
            parts = group.lower().split("/")
            if any(part in {"train", "student_test"} for part in parts):
                continue
            groups[group][trip_id].add(suffix)
    valid = {prefix: trips for prefix, trips in groups.items()
             if len(trips) == 10 and all(suffixes == REQUIRED for suffixes in trips.values())}
    if selected_group is None and len(valid) == 1:
        selected_group = next(iter(valid))
    if selected_group not in valid:
        raise ValueError("Expected exactly ten complete evaluator trips in one group. Use --group if needed.")
    return {"schema_version": "1.0", "mode": "evaluation", "partition": "evaluator_test",
            "dataset": str(dataset.resolve()), "dataset_prefix": selected_group, "flat_trip_root": True,
            "trip_ids": sorted(valid[selected_group]),
            "allowed_origins": ["http://localhost:8080", "http://127.0.0.1:8080"],
            "artifact_manifest": "configs/inference.final.json"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--group", help="directory in the ZIP or relative to the supplied root")
    parser.add_argument("--output", type=Path, default=Path("config.local.json"))
    args = parser.parse_args()
    settings = evaluator_config(args.dataset, args.group)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(settings, indent=2) + "\n")
    print(f"Configured {len(settings['trip_ids'])} evaluator choices in {args.output}")


if __name__ == "__main__":
    main()
