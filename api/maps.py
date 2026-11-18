from pathlib import Path

import folium

ASSETS = Path(__file__).resolve().parents[1] / "ui" / "vendor"


def offline_folium_map(gps):
    """An interactive map requiring no external scripts or basemap requests."""
    view = folium.Map(location=gps[0], tiles=None)
    view.default_js = []
    view.default_css = []
    stylesheet = (ASSETS / "leaflet.css").read_text(encoding="utf-8")
    script = (ASSETS / "leaflet.js").read_text(encoding="utf-8").split("//# sourceMappingURL=")[0]
    view.get_root().header.add_child(folium.Element(f"<style>{stylesheet}</style><script>{script}</script>"))
    view.fit_bounds(gps)
    return view


def exported_route_map(gps, segments):
    route_points = [point for segment in segments for point in segment]
    view = offline_folium_map(gps + route_points)
    folium.map.CustomPane("raw-gps", z_index=410).add_to(view)

    route_layer = folium.FeatureGroup(name="Predicted route").add_to(view)
    for segment in segments:
        folium.PolyLine(segment, color="#00877a", weight=6,
                        tooltip="Predicted route").add_to(route_layer)

    gps_layer = folium.FeatureGroup(name="Raw GPS").add_to(view)
    if len(gps) == 1:
        folium.CircleMarker(gps[0], pane="raw-gps", radius=5, color="#c4530b", fill=True,
                            fill_color="#ff9f43", fill_opacity=1,
                            tooltip="Raw GPS").add_to(gps_layer)
    else:
        folium.PolyLine(gps, pane="raw-gps", color="#ed7617", weight=4, dash_array="5 8",
                        tooltip="Raw GPS").add_to(gps_layer)
    folium.LayerControl().add_to(view)
    return view
