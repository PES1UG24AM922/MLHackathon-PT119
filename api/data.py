from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from src.eda.data_loader import TripData


class TripStore:
    def __init__(self, path: Path, prefix: str, partition: str, flat_trip_root=False):
        self.path = path
        self.prefix = prefix.strip("/")
        self.partition = partition
        directory = "" if flat_trip_root else partition
        self.root = "/".join(part for part in (self.prefix, directory) if part)

    def _read(self, trip_id, suffixes):
        names = [f"{trip_id}/{trip_id}.{suffix}" for suffix in suffixes]
        if self.root:
            names = [f"{self.root}/{name}" for name in names]
        if self.path.is_dir():
            return [(self.path / name).read_bytes() for name in names]
        with ZipFile(self.path) as archive:
            return [archive.read(name) for name in names]

    def load_unlabeled(self, trip_id: str) -> TripData:
        track_raw, nodes_raw, arcs_raw = self._read(trip_id, ("track", "nodes", "arcs"))
        track = self._table(track_raw, ["lon", "lat", "timestamp"])
        nodes = self._table(nodes_raw, ["lon", "lat"])
        arcs = self._table(arcs_raw, ["from_node", "to_node"])
        for frame in (track, nodes):
            if not (frame.lon.between(-180, 180).all() and frame.lat.between(-90, 90).all()):
                raise ValueError("Invalid geographic coordinates")
        arc_values = arcs.to_numpy()
        if (arc_values < 0).any() or (arc_values >= len(nodes)).any() or (arc_values % 1 != 0).any():
            raise ValueError("Invalid arc endpoints")
        return TripData(self.partition, trip_id, self.path, track, nodes, arcs.astype(int), None)

    @staticmethod
    def _table(raw, columns):
        frame = pd.read_csv(BytesIO(raw), sep=r"\s+", header=None, dtype=float)
        if frame.empty or len(frame.columns) != len(columns) or not np.isfinite(frame.to_numpy()).all():
            raise ValueError("Invalid trip table")
        frame.columns = columns
        return frame

    def read_reference(self, trip_id: str, n_arcs: int) -> list[int]:
        (raw,) = self._read(trip_id, ("route",))
        route = [int(value) for value in raw.split()]
        if not route or any(not 0 <= value < n_arcs for value in route):
            raise ValueError("Invalid reference route")
        return route
