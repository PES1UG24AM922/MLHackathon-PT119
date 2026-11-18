"""Extract only organizer student data and build the aligned candidate cache."""
import argparse
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from src.features.build_dataset import build_feature_datasets

TRIP_FILE = re.compile(r"Student folder/(train|student_test)/([0-9]{8})/\2\.(track|nodes|arcs|route)")
REQUIRED = {"track", "nodes", "arcs", "route"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive:
        audited = defaultdict(set)
        files = []
        for name in archive.namelist():
            match = TRIP_FILE.fullmatch(name)
            if match:
                split, trip_id, suffix = match.groups()
                if suffix in audited[(split, trip_id)]:
                    raise ValueError(f"Duplicate student input {name}")
                audited[(split, trip_id)].add(suffix)
                files.append(name)
        if any(suffixes != REQUIRED for suffixes in audited.values()):
            raise ValueError("Incomplete student trip")
        counts = {split: sum(key[0] == split for key in audited) for split in ("train", "student_test")}
        if counts != {"train": 80, "student_test": 10}:
            raise ValueError(f"Expected the organizer's student archive, got {counts}")
        for name in files:
            archive.extract(name, args.work)
    build_feature_datasets(base_dir=args.work, artifacts_dir=args.work / "cache",
                           results_dir="results/final")


if __name__ == "__main__":
    main()
