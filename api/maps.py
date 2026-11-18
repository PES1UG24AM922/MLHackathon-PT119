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
