"""Folium visualisation of actual classifier and decoder outputs."""

from __future__ import annotations

from typing import Iterable

from .graph import RoadGraph
from .schemas import DecodeResult, GPSPoint


def build_folium_map(
    graph: RoadGraph,
    gps_points: Iterable[GPSPoint],
    classifier_arc_ids: dict[int, int],
    decoded_result: DecodeResult,
    reference_arc_ids: Iterable[int] | None = None,
):
    """Return a Folium map; reference layer is for offline evaluation only."""
    try:
        import folium
    except ImportError as exc:  # pragma: no cover - dependency message only
        raise RuntimeError("Install folium to generate route maps: pip install folium") from exc

    points = list(gps_points)
    usable_latlon = [
        (point.latitude, point.longitude)
        for point in points
        if point.latitude is not None and point.longitude is not None
    ]
    if not usable_latlon:
        raise ValueError("Folium output requires longitude/latitude on GPS points")
    centre = [
        sum(latitude for latitude, _ in usable_latlon) / len(usable_latlon),
        sum(longitude for _, longitude in usable_latlon) / len(usable_latlon),
    ]
    route_map = folium.Map(location=centre, zoom_start=13, tiles="CartoDB positron")

    gps_layer = folium.FeatureGroup(name="Raw GPS observations", show=True)
    folium.PolyLine(usable_latlon, color="#2563eb", weight=2, opacity=0.7).add_to(gps_layer)
    gps_layer.add_to(route_map)

    classifier_layer = folium.FeatureGroup(name="Classifier-only route", show=False)
    _add_arcs(
        classifier_layer,
        graph,
        _compact(classifier_arc_ids[index] for index in sorted(classifier_arc_ids)),
        "#f59e0b",
        3,
    )
    classifier_layer.add_to(route_map)

    decoder_layer = folium.FeatureGroup(name="HMM/Viterbi predicted route", show=True)
    for segment in decoded_result.expanded_route_segments:
        _add_arcs(decoder_layer, graph, segment, "#7c3aed", 4)
    decoder_layer.add_to(route_map)

    if reference_arc_ids is not None:
        reference_layer = folium.FeatureGroup(name="Reference route (offline evaluation only)", show=False)
        _add_arcs(reference_layer, graph, _compact(reference_arc_ids), "#16a34a", 3)
        reference_layer.add_to(route_map)
    folium.LayerControl(collapsed=False).add_to(route_map)
    return route_map


def _add_arcs(layer, graph: RoadGraph, arc_ids: Iterable[int], color: str, weight: int) -> None:
    import folium

    for arc_id in arc_ids:
        if not graph.has_arc(int(arc_id)):
            continue
        folium.PolyLine(
            graph.arc_geometry_latlon(int(arc_id)),
            color=color,
            weight=weight,
            opacity=0.8,
            tooltip=f"arc_id={arc_id}",
        ).add_to(layer)


def _compact(arc_ids: Iterable[int]) -> list[int]:
    compact: list[int] = []
    for arc_id in arc_ids:
        if not compact or compact[-1] != arc_id:
            compact.append(int(arc_id))
    return compact
