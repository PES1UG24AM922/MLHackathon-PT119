"""Versioned, public inference contract. Coordinates use GeoJSON order: lon, lat."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

VERSION = "1.0"

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class PredictRequest(StrictModel):
    trip_id: str = Field(pattern=r"^[0-9]{8}$")

class Coordinate(StrictModel):
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)

class Prediction(StrictModel):
    schema_version: Literal["1.0"] = VERSION
    request_id: str
    trip_id: str
    mode: Literal["development", "evaluation"]
    model_version: str
    decoder_version: str
    score_kind: Literal["uncalibrated", "calibrated"]
    duration_ms: float = Field(ge=0)
    gps: list[Coordinate]
    route_segments: list[list[Coordinate]]
    predicted_arc_ids: list[int | None]
    warnings: list[str]
    metrics: dict[Literal[
        "ordered_route_recall", "ordered_route_precision", "ordered_route_f1", "point_top1_accuracy"
    ], float] = {}
