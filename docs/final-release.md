# Run the final PathSnap build

Use Python 3.12. From the repository root:

```powershell
python -m venv .venv
& .\.venv\Scripts\python -m pip install -e . -r requirements.txt
$env:PATHSNAP_DATASET = 'C:\path\to\Student folder.zip'
& .\.venv\Scripts\python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --no-access-log
```

Laptop 1 runs that API. On Laptop 2, open `http://LAPTOP_1_IP:8000/` on the
same LAN. No internet connection is needed for the UI or route exports.
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

## Reproduce the final experiment

```powershell
& .\.venv\Scripts\python -m scripts.prepare_student 'C:\path\to\Student folder.zip' --work ../pathsnap-student
& .\.venv\Scripts\python -m scripts.final_train --cache ../pathsnap-student/cache/feature_datasets.joblib --scratch ../pathsnap-student/cache --cap 80000 --trials 12
& .\.venv\Scripts\python -m scripts.model_continuity --data ../pathsnap-student --cache ../pathsnap-student/cache/feature_datasets.joblib
& .\.venv\Scripts\python -m scripts.final_decode --data ../pathsnap-student --cache ../pathsnap-student/cache/feature_datasets.joblib --scores ../pathsnap-student/cache/decoder_scores.npz
& .\.venv\Scripts\python -m pytest -q
```

`results/final/` contains the alignment-era comparisons, per-fold and
Bayesian trial logs, candidate misses, HMM comparison, and student-test
validation. The historical six-model comparison at `results/model_comparison.csv`
used earlier nearest-arc labels and has its own replay script. Scores with
different label versions are not head-to-head results.

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

To rebuild the consolidated PDF from the tracked evidence:

```powershell
& .\.venv\Scripts\python -m pip install -e '.[report]'
& .\.venv\Scripts\python -m scripts.submission_report PathSnap-technical-report.pdf
```
