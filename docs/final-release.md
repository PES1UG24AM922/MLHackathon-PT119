# Run the final PathSnap build

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
& .\.venv\Scripts\python -m pip install -e . -r requirements.txt
$env:PATHSNAP_DATASET = 'C:\path\to\Student folder.zip'
& .\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

For the extracted student folder on Fedora, start from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . -r requirements.txt
unset PATHSNAP_CONFIG
export PATHSNAP_DATASET="$PWD/.."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8002 --no-access-log
```

`PATHSNAP_DATASET` points to the parent of `Student folder`, which the student
configuration adds to the trip path. Use the same port in the browser and
acceptance commands as on the server.

Laptop 1 runs that API. On Laptop 2, open `http://LAPTOP_1_IP:8002/ui/` (or
port 8000 for the PowerShell example) on the same LAN. The street basemap
requires internet access from Laptop 2. The plain background, route overlays,
and downloaded Folium maps are available offline.
Select a trip, run reconstruction, then download its Folium map. If the page
does not load, check the server's `/ready` response and Laptop 1's inbound
firewall rule for TCP port 8000. Keep the API on the demonstration network.

## Official evaluation

Run these only once the organizer authorizes the reserved data:

```powershell
Remove-Item Env:PATHSNAP_DATASET -ErrorAction SilentlyContinue
& .\.venv\Scripts\python -m scripts.configure_demo 'C:\path\to\evaluator_test.zip'
$env:PATHSNAP_CONFIG = (Resolve-Path config.local.json).Path
& .\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

The setup discovers one group of ten complete trips in an archive or directory.
Use `--group` when there are several groups. The ignored local configuration
contains the organizer's path and allowed trip IDs. Keep the ZIP, extracted
files, evaluator results, and that configuration out of version control.

At the demo, confirm `/ready`, ten choices on Laptop 2, and the response and
terminal request ID for each choice. Capture the actual two-laptop evidence.
The model receives only track and network geometry. The API reads references
after prediction and returns only aggregate point Top-1 and ordered-route
scores. Student-test reference maps are offline report assets.

In a separate terminal, run the authorized checks against the live evaluator
server using the same port as Uvicorn:

```bash
python -m scripts.check_api --partition evaluator_test \
  --base-url http://127.0.0.1:8002 --output outputs-local/evaluator
```

This checks all ten live predictions and map responses, the UI assets, and
invalid requests. It writes `api_evaluator_test.csv` under the ignored output
directory. Evaluator map exports are checked in memory.

## Reproduce the final experiment

```powershell
& .\.venv\Scripts\python -m scripts.prepare_student 'C:\path\to\Student folder.zip' --work ../pathsnap-student
& .\.venv\Scripts\python -m scripts.final_train --cache ../pathsnap-student/cache/feature_datasets.joblib --scratch ../pathsnap-student/cache --cap 80000 --trials 12
& .\.venv\Scripts\python -m scripts.model_continuity --data ../pathsnap-student --cache ../pathsnap-student/cache/feature_datasets.joblib
& .\.venv\Scripts\python -m scripts.final_decode --data ../pathsnap-student --cache ../pathsnap-student/cache/feature_datasets.joblib --scores ../pathsnap-student/cache/decoder_scores.npz
& .\.venv\Scripts\python -m pytest -q
```

`prepare_student` also accepts an extracted directory, such as `../Student folder`.
For that form, use `--data "../Student folder"` in the continuity and decoding
commands and keep the cache path under the chosen `--work` directory.

`results/final/` contains the alignment-era comparisons, per-fold and
Bayesian trial logs, candidate misses, HMM comparison, and student-test
validation. The historical six-model comparison at `results/model_comparison.csv`
used earlier nearest-arc labels. To replay it, check out `5d2bd93` in a separate
worktree, place the organizer's extracted `Student folder` beside `src/`, and run
`python -m src.models.train_compare` with that revision's dependencies.

Five GroupKFold splits isolate training and validation trips. Estimator fits
use at most 80,000 stratified candidate rows per fit to bound runtime. Validation
and student scoring cover every observation and candidate. Train-only Top-1
chooses the model and Bayesian configuration. Outer held-out decoder scores
come from models and calibrators that did not see those trips. Decoder
parameter selection uses two deterministic length-ranked trips from each
outer fold. The final decoder comparison uses all ten student-test trips.

The live API smoke check can be repeated against the local student server:

```powershell
& .\.venv\Scripts\python -m scripts.check_api --base-url http://127.0.0.1:8000 --output outputs-local
```

The consolidated technical report is at `output/pdf/PathSnap-technical-report.pdf`.
The report uses the archived release runs for model selection, Bayesian trials,
and HMM reconstruction. It also uses a separate seven-model diagnostic replay
on the same student partitions. The replay includes observation-level Macro F1,
observation-level log loss, and binary candidate confusion counts at threshold
0.5.

To reproduce the report diagnostics after preparing the student data:

```powershell
& .\.venv\Scripts\python -m scripts.submission_metrics --data ../pathsnap-student --cache ../pathsnap-student/cache/feature_datasets.joblib --out outputs-local/diagnostics
```

Use the extracted student directory for `--data` if it was passed to `prepare_student`.
The four diagnostic CSVs are in `results/final/`. The candidate-miss logs retain
both the original release run and the report replay.

To rebuild it from the tracked evidence:

```powershell
& .\.venv\Scripts\python -m pip install -e '.[report]'
& .\.venv\Scripts\python -m scripts.submission_report output/pdf/PathSnap-technical-report.pdf
```

To reproduce the supplementary MLP training/validation loss curve, run
`python -m scripts.mlp_validation_history --data ../pathsnap-student` after
preparing the student data. The report reads the two committed `demo_*.png`
captures in `results/final/`. To update the second-laptop screenshot, pass
`--api-screenshot PATH_TO_CAPTURE.png` when rebuilding the report.
