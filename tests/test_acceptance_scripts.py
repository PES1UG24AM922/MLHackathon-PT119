import csv
import json

import httpx
import pytest

from scripts import check_api, prepare_student


@pytest.mark.parametrize(("partition", "mode"), [
    ("student_test", "development"),
    ("evaluator_test", "evaluation"),
])
def test_checker_saves_map_only_for_student_runs(tmp_path, monkeypatch, partition, mode):
    ids = [f"{index:08d}" for index in range(10)]

    def reply(request):
        path = request.url.path
        if path == "/predictions":
            trip = json.loads(request.content)["trip_id"]
            if trip == "../":
                return httpx.Response(422)
            if trip not in ids:
                return httpx.Response(404)
            return httpx.Response(200, headers={"X-Request-ID": trip}, json={
                "request_id": trip, "trip_id": trip, "mode": mode, "gps": [{}],
                "predicted_arc_ids": [0], "route_segments": [[{}]], "duration_ms": 1,
                "model_version": "test", "decoder_version": "test",
                "metrics": dict.fromkeys(check_api.METRICS, 1),
            })
        if path == "/maps/missing":
            return httpx.Response(404)
        if path.startswith("/maps/"):
            return httpx.Response(200, text="Raw GPS / Predicted route")
        return {
            "/health": httpx.Response(200, json={"mode": mode}),
            "/ready": httpx.Response(200, json={"score_kind": "calibrated"}),
            "/trips": httpx.Response(200, json={"partition": partition, "trip_ids": ids}),
            "/ui/": httpx.Response(200, text="Reconstruct route"),
            "/ui/vendor/leaflet.js": httpx.Response(200),
        }[path]

    client_class = httpx.Client
    monkeypatch.setattr(check_api.httpx, "Client", lambda **kwargs: client_class(
        transport=httpx.MockTransport(reply), **kwargs
    ))
    check_api.run("http://test", tmp_path, partition)
    with (tmp_path / f"api_{partition}.csv").open(newline="") as evidence:
        rows = list(csv.DictReader(evidence))
    assert [row["trip_id"] for row in rows] == ids
    assert {row["check"] for row in rows} == {"pass"}
    assert bool(list(tmp_path.glob("*.html"))) == (mode == "development")


def test_student_cache_excludes_evaluator_paths():
    names = [
        f"Student folder/{split}/{index:08d}/{index:08d}.{suffix}"
        for split, indices in (("train", range(80)), ("student_test", range(80, 90)),
                               ("evaluator_test", range(90, 100)))
        for index in indices
        for suffix in prepare_student.REQUIRED
    ]
    selected = prepare_student.student_files(names)
    assert len(selected) == 360
    assert all("/evaluator_test/" not in name for name in selected)
    with pytest.raises(ValueError, match="Incomplete student trip"):
        prepare_student.student_files(names[1:])
