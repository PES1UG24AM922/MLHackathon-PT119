import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from api.main import ROOT, create_app


def test_reference_does_not_change_real_inference(tmp_path, monkeypatch):
    monkeypatch.delenv("PATHSNAP_DATASET", raising=False)
    archive = tmp_path / "trip.zip"
    trip_id = "00000000"
    prefix = f"Student folder/student_test/{trip_id}/{trip_id}"
    with ZipFile(archive, "w") as source:
        source.writestr(prefix + ".track", "55 56 0\n55.0001 56 1\n")
        source.writestr(prefix + ".nodes", "55 56\n55.0002 56\n")
        source.writestr(prefix + ".arcs", "0 1\n")
    cfg = json.loads((ROOT / "configs/api.student.json").read_text())
    cfg["dataset"] = str(archive)
    with TestClient(create_app(cfg)) as client:
        assert client.get("/ready").status_code == 200
        preflight = client.options("/predictions", headers={
            "Origin": "http://localhost:8080", "Access-Control-Request-Method": "POST",
        })
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:8080"
        body = {"trip_id": trip_id}
        first = client.post("/predictions", json=body).raise_for_status().json()
        assert first["predicted_arc_ids"] == [0, 0]
        assert not first["metrics"]
        assert "REFERENCE_METRICS_UNAVAILABLE" in first["warnings"]
        with ZipFile(archive, "a") as source:
            source.writestr(prefix + ".route", "0\n")
        second = client.post("/predictions", json=body).raise_for_status().json()
        assert second["predicted_arc_ids"] == first["predicted_arc_ids"]
        assert second["route_segments"] == first["route_segments"]
        assert second["metrics"]["ordered_route_f1"] == 1.0
        export = client.get(f"/maps/{second['request_id']}").raise_for_status()
        assert "Predicted route" in export.text
        assert "Reference route" not in export.text
        invalid = client.post("/predictions", json={**body, "labels": "not echoed"})
        assert invalid.status_code == 422
        assert "not echoed" not in invalid.text


def test_manifest_failure_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.delenv("PATHSNAP_DATASET", raising=False)
    cfg = json.loads((ROOT / "configs/api.student.json").read_text())
    cfg.update(dataset=str(tmp_path), artifact_manifest=str(tmp_path / "missing.json"))
    with TestClient(create_app(cfg)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predictions", json={"trip_id": "00000000"}).status_code == 503
