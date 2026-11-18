"""Reading raw PathSnap trip folders.

Each trip folder <id>/ contains:
  <id>.nodes  lon lat               (row index = node id)
  <id>.arcs   from_node to_node     (row index = arc id, directed)
  <id>.track  lon lat t_seconds     (GPS observations, ~1 Hz)
  <id>.route  arc_id                (reference route in driving order; may be absent)
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    nodes: np.ndarray            # (N, 2) float  lon, lat
    arcs: np.ndarray             # (M, 2) int    from_node, to_node
    track: np.ndarray            # (T, 3) float  lon, lat, t
    route: np.ndarray | None     # (R,)   int    arc ids, or None if not supplied

    @property
    def n_points(self) -> int:
        return len(self.track)


def _read(path: Path, dtype, ncols: int) -> np.ndarray:
    arr = np.loadtxt(path, dtype=dtype, ndmin=2)
    if arr.size == 0:
        return np.empty((0, ncols), dtype=dtype)
    if arr.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, got {arr.shape[1]}")
    return arr


def load_trip(trip_dir: str | Path, with_route: bool = True) -> Trip:
    """Load one trip folder. Set with_route=False in the prediction path so the
    reference route can never leak into predictions."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    base = trip_dir / tid
    nodes = _read(base.with_suffix(".nodes"), float, 2)
    arcs = _read(base.with_suffix(".arcs"), int, 2)
    track = _read(base.with_suffix(".track"), float, 3)

    route = None
    route_path = base.with_suffix(".route")
    if with_route and route_path.exists():
        route = _read(route_path, int, 1).ravel()

    return Trip(tid, nodes, arcs, track, route)


def list_trips(split_dir: str | Path) -> list[Path]:
    """All trip folders inside a split directory (e.g. data/train), sorted by id."""
    return sorted(p for p in Path(split_dir).iterdir() if p.is_dir())
