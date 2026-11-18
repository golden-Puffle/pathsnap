"""Geometry helpers: lon/lat -> local metres, and point-to-segment distances.

Each trip covers a few km, so a local equirectangular projection centred on the
trip is accurate to well under 1 m at that scale and is much cheaper than full
geodesic maths. Degrees must never be used directly as distances.
"""
import numpy as np

EARTH_RADIUS_M = 6_371_008.8


class LocalProjector:
    """Equirectangular projection around a reference point (lon0, lat0)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = lon0
        self.lat0 = lat0
        self.kx = np.radians(1.0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
        self.ky = np.radians(1.0) * EARTH_RADIUS_M

    @classmethod
    def for_points(cls, lonlat: np.ndarray) -> "LocalProjector":
        lon0, lat0 = np.mean(lonlat[:, 0]), np.mean(lonlat[:, 1])
        return cls(lon0, lat0)

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=float)
        x = (lonlat[..., 0] - self.lon0) * self.kx
        y = (lonlat[..., 1] - self.lat0) * self.ky
        return np.stack([x, y], axis=-1)

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        lon = xy[..., 0] / self.kx + self.lon0
        lat = xy[..., 1] / self.ky + self.lat0
        return np.stack([lon, lat], axis=-1)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used to sanity-check the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Distance from points p to segments a->b (all (..., 2) in metres, broadcastable).

    Returns (distance, t) where t in [0, 1] is the position of the closest point
    along the segment (0 = at a, 1 = at b).
    """
    ab = b - a
    ap = p - a
    denom = np.einsum("...i,...i->...", ab, ab)
    t = np.where(denom > 0, np.einsum("...i,...i->...", ap, ab) / np.where(denom > 0, denom, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1), t


def bearing_deg(dx, dy):
    """Compass bearing in degrees (0 = north, 90 = east) of a vector in metres."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)
