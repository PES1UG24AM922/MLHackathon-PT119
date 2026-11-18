from zipfile import ZipFile

from api.data import TripStore
from scripts.configure_demo import evaluator_config


def test_flat_evaluator_layout_uses_only_supplied_whitelist(tmp_path):
    archive = tmp_path / "evaluator.zip"
    with ZipFile(archive, "w") as data:
        for i in range(10):
            trip_id = f"{100 + i:08d}"
            for ext in ("track", "nodes", "arcs", "route"):
                data.writestr(f"provided/evaluation/{trip_id}/{trip_id}.{ext}", "0")
    settings = evaluator_config(archive)
    assert settings["dataset_prefix"] == "provided/evaluation"
    assert settings["trip_ids"][0] == "00000100"
    store = TripStore(archive, settings["dataset_prefix"], settings["partition"],
                      flat_trip_root=settings["flat_trip_root"])
    assert store.read_reference("00000100", n_arcs=1) == [0]
