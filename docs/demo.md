# PathSnap two-laptop demo

This page describes the original Person 4 prototype. Use the [final release
guide](final-release.md) for the calibrated model and the current demo configuration.

The development configuration runs the committed Random Forest and Person 3 decoder against the ten student-test trips. Candidate scores from this model are uncalibrated. Evaluator mode requires the final calibrated artifact.

## Laptop 1

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e . -r requirements.txt
$env:PATHSNAP_DATASET = 'C:\path\to\Student folderhvi.zip'
.\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

Confirm `/ready` returns 200. Keep the terminal open for request logs. Endpoint schemas are available at `/docs`.

## Laptop 2

On the same local network, open `http://LAPTOP_1_IP:8000/`. Select a trip and choose **Reconstruct route**. The UI and standalone Folium export render GPS and route overlays using bundled Leaflet assets. They use a plain map background and work without internet.

To host the UI separately on Laptop 2, serve only the `ui/` folder with `python -m http.server 8080 --directory ui`. Open `http://localhost:8080`, enter Laptop 1's API address and choose **Connect**. The provided CORS configuration allows localhost port 8080. Add the actual UI origin if using another address.

## Evaluation and artifact handoff

Copy `configs/api.student.json` to ignored `config.local.json`. Set `mode` to `evaluation` and configure the organizer's partition, prefix, data path, ten reserved IDs, and final artifact manifest. Point `PATHSNAP_CONFIG` at the new file and set or clear `PATHSNAP_DATASET` as needed. Folder and ZIP inputs use `[prefix/]partition/trip_id/trip_id.extension`.

The final model bundle uses `model`, `needs_scaling`, `scaler`, and ordered `feature_cols`. The model implements `predict_proba` with positive class 1. Update the model, decoder config, versions, score kind, scikit-learn pin, and hashes together. The decoder config hash uses LF line endings. Keep organizer evaluator files outside Git.

Prediction loads track, nodes, and arcs. The separate evaluator accesses the ordered reference after inference and returns three aggregate scores. It computes the longest common subsequence of predicted and reference arc IDs, preserving order and repeated arcs. Recall divides the common count by reference length. Precision divides it by predicted length. F1 uses both lengths. Point-level accuracy requires separately aligned labels.

## Acceptance

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python scripts/check_api.py --base-url http://127.0.0.1:8000
```

The HTTP check selects every student trip and downloads each Folium map. The integration tests check the real scorer and decoder with and without reference data. For final acceptance, repeat all ten evaluator selections across the actual two laptops. Record restart, latency, loading state, rendering, exports, repeated requests, warning display, and reference isolation.
