"""Run acceptance checks against a PathSnap student or final evaluator server."""
import argparse
import csv
from pathlib import Path

import httpx


MODES = {"student_test": "development", "evaluator_test": "evaluation"}
METRICS = {"ordered_route_recall", "ordered_route_precision",
           "ordered_route_f1", "point_top1_accuracy"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(base_url, evidence, partition="student_test"):
    evidence.mkdir(parents=True, exist_ok=True)
    rows = []
    mode = MODES[partition]
    with httpx.Client(base_url=base_url, timeout=300) as client:
        health = client.get("/health").raise_for_status().json()
        readiness = client.get("/ready").raise_for_status().json()
        trips = client.get("/trips").raise_for_status().json()
        require(health["mode"] == mode and trips["partition"] == partition,
                f"Expected a server configured for {partition}")
        if mode == "evaluation":
            require(readiness["score_kind"] == "calibrated", "Expected a calibrated evaluator scorer")
        require(len(trips["trip_ids"]) == len(set(trips["trip_ids"])) == 10,
                "Expected ten distinct trip IDs")
        for trip_id in trips["trip_ids"]:
            row = {"trip_id": trip_id}
            try:
                response = client.post("/predictions", json={"trip_id": trip_id})
                row.update(status=response.status_code, request_id=response.headers.get("X-Request-ID"))
                response.raise_for_status()
                result = response.json()
                require(bool(row["request_id"]) and result["request_id"] == row["request_id"],
                        "Missing or mismatched request ID")
                require(result["trip_id"] == trip_id and result["mode"] == mode,
                        "Trip ID or mode mismatched")
                require(len(result["predicted_arc_ids"]) == len(result["gps"]) > 0,
                        "Missing or misaligned point predictions")
                require(bool(result["route_segments"]), "No route geometry returned")
                require(result["metrics"].keys() == METRICS, "Expected all four aggregate metrics")
                require(all(0 <= value <= 1 for value in result["metrics"].values()),
                        "Metric outside [0, 1]")
                require(all("reference" not in key.lower() for key in result),
                        "Response exposed a reference field")
                download = client.get(f"/maps/{row['request_id']}").raise_for_status()
                require(all(label in download.text for label in ("Raw GPS", "Predicted route")),
                        "Map export is missing a route layer")
                require("reference route" not in download.text.lower(),
                        "Map export exposed a reference layer")
                if mode == "development" and not rows:
                    (evidence / f"pathsnap-{trip_id}.html").write_text(download.text, encoding="utf-8")
                row.update(duration_ms=round(result["duration_ms"], 2), gps_points=len(result["gps"]),
                           segments=len(result["route_segments"]), **result["metrics"],
                           model_version=result["model_version"], decoder_version=result["decoder_version"])
                row["check"] = "pass"
            except (ValueError, KeyError, TypeError, httpx.HTTPError) as error:
                row.update(check="fail", error=str(error))
            print(row, flush=True)
            rows.append(row)
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (evidence / f"api_{partition}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        unknown_trip = next(f"{i:08d}" for i in range(99_999_999, 99_999_988, -1)
                            if f"{i:08d}" not in trips["trip_ids"])
        require(client.post("/predictions", json={"trip_id": unknown_trip}).status_code == 404,
                "Unknown trip was not rejected")
        require(client.post("/predictions", json={"trip_id": "../"}).status_code == 422,
                "Malformed trip ID was not rejected")
        require(client.get("/maps/missing").status_code == 404, "Missing map did not return 404")
        source = client.get("/ui/").raise_for_status()
        require("Reconstruct route" in source.text, "Route UI did not load")
        client.get("/ui/vendor/leaflet.js").raise_for_status()
        failures = [row["trip_id"] for row in rows if row["check"] == "fail"]
        if failures:
            raise RuntimeError(f"Failed trip checks: {failures}")
        print(f"{len(rows)}/10 {partition} trips passed. Invalid trip, missing map, and UI checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=Path("outputs-local"))
    parser.add_argument("--partition", choices=MODES, default="student_test",
                        help="Use evaluator_test only for the authorized final run")
    args = parser.parse_args()
    run(args.base_url, args.output, args.partition)
