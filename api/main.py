import json
import logging
import os
import re
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import BoundedSemaphore, Lock
from uuid import uuid4
from zipfile import BadZipFile

import folium
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .contracts import Prediction, PredictRequest
from .data import TripStore
from .evaluate import evaluate_prediction
from .pipeline import TeamPipeline

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("uvicorn.error")


def create_app(cfg=None):
    if cfg is None:
        path = Path(os.environ.get("PATHSNAP_CONFIG", ROOT / "configs" / "api.student.json"))
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    mode, partition, trip_ids = cfg["mode"], cfg["partition"], cfg["trip_ids"]
    if cfg["schema_version"] != "1.0" or mode not in {"development", "evaluation"}:
        raise ValueError("Invalid API configuration version or mode")
    if (mode == "development") != (partition == "student_test"):
        raise ValueError("Use student_test only with development mode")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", partition) or partition.lower().startswith("train"):
        raise ValueError("Invalid trip partition")
    prefix = cfg["dataset_prefix"]
    if prefix and any(not part or part in {".", ".."} or "\\" in part or ":" in part for part in prefix.split("/")):
        raise ValueError("Invalid dataset prefix")
    if len(set(trip_ids)) != 10 or len(trip_ids) != 10 or any(not re.fullmatch(r"[0-9]{8}", t) for t in trip_ids):
        raise ValueError("Provide ten distinct eight-digit trip IDs")
    dataset = Path(os.environ.get("PATHSNAP_DATASET", ROOT / cfg["dataset"]))
    store = TripStore(dataset, cfg["dataset_prefix"], partition)
    pipeline = None
    reason = "Pipeline is starting"
    slots = BoundedSemaphore(1)
    cache = OrderedDict()
    cache_lock = Lock()

    @asynccontextmanager
    async def lifespan(_app):
        nonlocal pipeline, reason
        with cache_lock:
            cache.clear()
        try:
            if not dataset.exists():
                raise FileNotFoundError("Dataset unavailable. Set PATHSNAP_DATASET.")
            loaded = TeamPipeline(ROOT, ROOT / cfg["artifact_manifest"])
            if mode == "evaluation" and loaded.manifest.score_kind != "calibrated":
                raise ValueError("Evaluator configuration requires the final calibrated scorer")
            pipeline = loaded
            reason = ""
        except Exception as error:
            reason = f"PathSnap startup: {error}"
            LOG.error("%s", reason)
        yield
        pipeline = None

    app = FastAPI(title="PathSnap", version="1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=cfg["allowed_origins"],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _exc):
        return JSONResponse({"detail": "Invalid request. Supply only an eight-digit trip_id."}, status_code=422)

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        request.state.request_id = uuid4().hex
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        finally:
            LOG.info("%s", json.dumps({
                "request_id": request.state.request_id, "status": status,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                "trip_id": getattr(request.state, "trip_id", None),
                "model_version": pipeline.manifest.model_version if pipeline else None,
                "decoder_version": pipeline.manifest.decoder_version if pipeline else None,
                "warnings": getattr(request.state, "warnings", []),
            }))

    @app.get("/health")
    def health():
        return {"status": "alive", "schema_version": "1.0", "mode": mode}

    @app.get("/ready")
    def ready():
        if pipeline is None:
            raise HTTPException(503, "Pipeline unavailable. Check Laptop 1 startup logs.")
        return {"status": "ready", "score_kind": pipeline.manifest.score_kind}

    @app.get("/trips")
    def trips():
        return {"partition": partition, "trip_ids": trip_ids}

    @app.post("/predictions", response_model=Prediction)
    def predict(body: PredictRequest, request: Request):
        if body.trip_id not in trip_ids:
            raise HTTPException(404, "Trip is not permitted")
        request.state.trip_id = body.trip_id
        if pipeline is None:
            raise HTTPException(503, "Pipeline unavailable. Check Laptop 1 startup logs.")
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "Another trip is processing. Try again shortly.")
        start = time.perf_counter()
        try:
            try:
                trip = store.load_unlabeled(body.trip_id)
            except (OSError, KeyError, ValueError, BadZipFile):
                raise HTTPException(422, "Required trip inputs are missing or invalid") from None
            point_arcs, segments, arc_segments, warnings = pipeline.predict_trip(trip)
            metrics = {}
            try:
                metrics = evaluate_prediction(store, body.trip_id, len(trip.arcs), arc_segments)
            except (OSError, KeyError, ValueError, BadZipFile):
                warnings.append("REFERENCE_METRICS_UNAVAILABLE")
            request.state.warnings = warnings
            result = Prediction(
                request_id=request.state.request_id, trip_id=body.trip_id, mode=mode,
                model_version=pipeline.manifest.model_version,
                decoder_version=pipeline.manifest.decoder_version,
                score_kind=pipeline.manifest.score_kind,
                duration_ms=(time.perf_counter() - start) * 1000,
                gps=trip.track[["lon", "lat"]].to_dict(orient="records"), route_segments=segments,
                predicted_arc_ids=point_arcs, warnings=warnings, metrics=metrics,
            )
            with cache_lock:
                cache[result.request_id] = result
                if len(cache) > 12:
                    cache.popitem(last=False)
            return result
        except HTTPException:
            raise
        except Exception as error:
            LOG.error("Prediction %s failed (%s)", request.state.request_id, type(error).__name__)
            raise HTTPException(500, "Inference failed. Share the request ID with the operator.") from None
        finally:
            slots.release()

    @app.get("/maps/{prediction_id}", response_class=HTMLResponse)
    def map_export(prediction_id: str, request: Request):
        with cache_lock:
            result = cache.get(prediction_id)
        if result is None:
            raise HTTPException(404, "Map result expired. Reconstruct the trip again.")
        request.state.trip_id = result.trip_id
        gps = [[p.lat, p.lon] for p in result.gps]
        view = folium.Map(location=gps[0], tiles=None)
        view.default_js = []
        view.default_css = []
        assets = ROOT / "ui" / "vendor"
        stylesheet = (assets / "leaflet.css").read_text(encoding="utf-8")
        script = (assets / "leaflet.js").read_text(encoding="utf-8").split("//# sourceMappingURL=")[0]
        view.get_root().header.add_child(folium.Element(f"<style>{stylesheet}</style><script>{script}</script>"))
        folium.PolyLine(gps, color="#667990", weight=2, tooltip="Raw GPS").add_to(view)
        for segment in result.route_segments:
            folium.PolyLine([[p.lat, p.lon] for p in segment], color="#008878", weight=4,
                            tooltip="Predicted route").add_to(view)
        view.fit_bounds(gps)
        return HTMLResponse(view.get_root().render(), headers={
            "Content-Disposition": f'attachment; filename="pathsnap-{result.trip_id}.html"'
        })

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/ui/")

    app.mount("/ui", StaticFiles(directory=ROOT / "ui", html=True), name="ui")
    return app


app = create_app()
