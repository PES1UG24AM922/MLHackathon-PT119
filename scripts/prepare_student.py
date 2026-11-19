"""Build the aligned candidate cache from a student archive or extracted folder."""
import argparse
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from src.eda.data_loader import resolve_data_root
from src.features.build_dataset import build_feature_datasets

TRIP_FILE = re.compile(r"Student folder/(train|student_test)/([0-9]{8})/\2\.(track|nodes|arcs|route)")
REQUIRED = {"track", "nodes", "arcs", "route"}


def student_files(names):
    audited = defaultdict(set)
    files = []
    for name in names:
        match = TRIP_FILE.fullmatch(name)
        if match is None:
            continue
        split, trip_id, suffix = match.groups()
        if suffix in audited[(split, trip_id)]:
            raise ValueError(f"Duplicate student input {name}")
        audited[(split, trip_id)].add(suffix)
        files.append(name)
    if any(suffixes != REQUIRED for suffixes in audited.values()):
        raise ValueError("Incomplete student trip")
    counts = {split: sum(key[0] == split for key in audited) for split in ("train", "student_test")}
    if counts != {"train": 80, "student_test": 10}:
        raise ValueError(f"Expected the organizer's student partitions, got {counts}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="student ZIP or directory containing the student partitions")
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/final"))
    args = parser.parse_args()
    if args.source.is_dir():
        data_root = resolve_data_root(args.source)
        student_files(
            f"Student folder/{path.relative_to(data_root).as_posix()}"
            for split in ("train", "student_test")
            for path in (data_root / split).glob("*/*")
            if path.is_file()
        )
    else:
        data_root = args.work
        with ZipFile(args.source) as archive:
            for name in student_files(archive.namelist()):
                archive.extract(name, args.work)
    build_feature_datasets(base_dir=data_root, artifacts_dir=args.work / "cache",
                           results_dir=args.out)


if __name__ == "__main__":
    main()
