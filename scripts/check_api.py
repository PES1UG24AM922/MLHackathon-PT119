"""Run the student-test acceptance checks against a running PathSnap service."""
import argparse
import csv
from pathlib import Path

import httpx


def run(base_url, evidence):
    evidence.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    with httpx.Client(base_url=base_url, timeout=300) as client:
        health = client.get("/health").raise_for_status().json()
        client.get("/ready").raise_for_status()
        trips = client.get("/trips").raise_for_status().json()
        if health["mode"] != "development" or trips["partition"] != "student_test":
            raise ValueError("This check records evidence only for the student-test partition")
        assert len(trips["trip_ids"]) == 10
        for trip_id in trips["trip_ids"]:
            response = client.post("/predictions", json={"trip_id": trip_id})
            result = response.json()
            row = {"trip_id": trip_id, "status": response.status_code,
                   "request_id": response.headers.get("X-Request-ID")}
            try:
                response.raise_for_status()
                assert result["request_id"] == row["request_id"]
                assert len(result["predicted_arc_ids"]) == len(result["gps"])
                assert result["route_segments"]
                assert result["mode"] == "development"
                expected = {"ordered_route_recall", "ordered_route_precision", "ordered_route_f1",
                            "point_top1_accuracy"}
                assert result["metrics"].keys() == expected
                assert all(0 <= value <= 1 for value in result["metrics"].values())
                assert all("reference" not in key for key in result)
                download = client.get(f"/maps/{row['request_id']}").raise_for_status()
                assert all(label in download.text for label in ("Raw GPS", "Predicted route"))
                assert "Reference route" not in download.text
                if not rows:
                    (evidence / f"pathsnap-{trip_id}.html").write_text(download.text, encoding="utf-8")
                row.update(duration_ms=round(result["duration_ms"], 2), gps_points=len(result["gps"]),
                           segments=len(result["route_segments"]), **result["metrics"])
                row["check"] = "pass"
            except (AssertionError, httpx.HTTPError) as error:
                failures.append(trip_id)
                row.update(check="fail", error=type(error).__name__)
            print(row, flush=True)
            rows.append(row)
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (evidence / "api_student_test.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        assert client.post("/predictions", json={"trip_id": "99999999"}).status_code == 404
        assert client.post("/predictions", json={"trip_id": "../"}).status_code == 422
        assert client.get("/maps/missing").status_code == 404
        source = client.get("/ui/").raise_for_status()
        assert "Reconstruct route" in source.text
        client.get("/ui/vendor/leaflet.js").raise_for_status()
        if failures:
            raise RuntimeError(f"Failed trip checks: {failures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("outputs-local"))
    args = parser.parse_args()
    run(args.base_url, args.output)
