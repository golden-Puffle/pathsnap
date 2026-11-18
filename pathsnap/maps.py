
"""Folium map helpers."""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjector
from .io import Trip


def _ll(xy_lonlat):
    """Convert (N, 2) longitude/latitude coordinates to Folium [lat, lon]."""
    return [[float(la), float(lo)] for lo, la in xy_lonlat]


def eda_trip_map(trip: Trip, out_html, partition: str = "", net_radius_m: float = 300.0):
    """Interactive map: raw GPS, nearby road network and reference route (if present)."""
    import folium

    if len(trip.track) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no GPS observations.")

    if len(trip.nodes) == 0:
        raise ValueError(f"Trip {trip.trip_id} has no road-network nodes.")

    proj = LocalProjector.for_points(trip.track[:, :2])

    xy_pts = proj.to_xy(trip.track[:, :2])
    xy_nodes = proj.to_xy(trip.nodes)

    u, v = trip.arcs[:, 0], trip.arcs[:, 1]

    # Keep only roads near the GPS trajectory to reduce HTML size.
    if len(trip.arcs) > 0:
        nearby_lists = cKDTree(xy_nodes).query_ball_point(
            xy_pts, r=net_radius_m
        )

        nonempty = [
            np.asarray(indices, dtype=int)
            for indices in nearby_lists
            if len(indices) > 0
        ]

        if nonempty:
            near = np.unique(np.concatenate(nonempty))
        else:
            near = np.array([], dtype=int)

        mask = np.zeros(len(xy_nodes), dtype=bool)

        if near.size > 0:
            mask[near] = True

        # Only draw arcs having at least one endpoint near the trajectory.
        valid_arcs = (
            (u >= 0)
            & (u < len(xy_nodes))
            & (v >= 0)
            & (v < len(xy_nodes))
        )

        sel = np.flatnonzero(
            valid_arcs & (mask[np.clip(u, 0, len(mask) - 1)] |
                          mask[np.clip(v, 0, len(mask) - 1)])
        )
    else:
        sel = np.array([], dtype=int)

    # Create map centered on the GPS trajectory.
    c = trip.track[:, :2].mean(axis=0)

    m = folium.Map(
        location=[c[1], c[0]],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Road network layer.
    net = folium.FeatureGroup(
        name=f"Road network (arcs within {int(net_radius_m)} m of GPS)"
    )

    for i in sel:
        folium.PolyLine(
            _ll(trip.nodes[[u[i], v[i]]]),
            color="#7f7f7f",
            weight=2,
            opacity=0.6
        ).add_to(net)

    net.add_to(m)

    # Reference route layer.
    if trip.route is not None:
        rt = folium.FeatureGroup(name="Reference route")

        valid_route = trip.route[
            (trip.route >= 0) & (trip.route < len(trip.arcs))
        ]

        for arc_id in valid_route:
            arc = trip.arcs[arc_id]
            from_node, to_node = arc

            if (
                0 <= from_node < len(trip.nodes)
                and 0 <= to_node < len(trip.nodes)
            ):
                folium.PolyLine(
                    _ll(trip.nodes[[from_node, to_node]]),
                    color="#1a9850",
                    weight=5,
                    opacity=0.85,
                    tooltip="Reference route"
                ).add_to(rt)

        rt.add_to(m)

    # Raw GPS observations layer.
    gps = folium.FeatureGroup(name="Raw GPS observations")

    for i, (lo, la, t) in enumerate(trip.track):
        folium.CircleMarker(
            [la, lo],
            radius=3,
            color="#1f3fff",
            fill=True,
            fill_opacity=0.9,
            weight=1,
            tooltip=f"obs {i} | t={t:.0f}s"
        ).add_to(gps)

    gps.add_to(m)

    # Start and end markers.
    se = folium.FeatureGroup(name="Start / End")

    folium.Marker(
        [trip.track[0, 1], trip.track[0, 0]],
        tooltip="Start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(se)

    folium.Marker(
        [trip.track[-1, 1], trip.track[-1, 0]],
        tooltip="End",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(se)

    se.add_to(m)

    # Map legend.
    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px;
                z-index: 9999; background: white;
                padding: 8px 12px; border: 1px solid #888;
                font-size: 13px;">
      <b>Trip {trip.trip_id}</b>
      {('(' + partition + ')') if partition else ''}<br>
      <span style="color:#1f3fff">&#9679;</span>
      Raw GPS observations<br>
      <span style="color:#1a9850">&#9644;</span>
      Reference route<br>
      <span style="color:#7f7f7f">&#9644;</span>
      Road network (near trajectory)
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend))

    folium.LayerControl(collapsed=False).add_to(m)

    # Fit map to GPS trajectory bounds.
    m.fit_bounds([
        [trip.track[:, 1].min(), trip.track[:, 0].min()],
        [trip.track[:, 1].max(), trip.track[:, 0].max()]
    ])

    # Save interactive HTML map.
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)

    m.save(str(out_html))

    return len(sel)
