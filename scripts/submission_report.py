"""Build the consolidated report from the committed results and demo captures."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image as PixelImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RESULTS = Path("results/final")
DARK = colors.HexColor("#16283d")
ACCENT = colors.HexColor("#007e72")
MUTED = colors.HexColor("#536477")
PAPER = colors.HexColor("#eff4f7")
WIDTH = 492

FAMILIES = {
    "GaussianNB": ("Probabilistic", "A Gaussian model for each class provides a fast baseline. "
                   "Correlated distance and rank features violate its independence assumption.",
                   "Maximum-likelihood class-conditional parameters"),
    "LogisticRegression": ("Linear", "The standardized linear model tests whether distance, direction "
                           "and candidate rank give a simple decision boundary. It cannot express many threshold interactions.",
                           "L2-regularized binary log loss"),
    "RandomForest": ("Bagged trees", "Tree splits capture interactions and directional thresholds. "
                     "Aggregating many trees handles noise but increases model size and inference work.",
                     "Gini impurity reduction at each split"),
    "HistGradientBoosting": ("Boosted trees", "Histogram splits capture nonlinear spatial and temporal features "
                             "without enumerating all thresholds. Small leaves and regularization limit complexity.",
                             "Binary log loss over additive trees"),
    "MLP": ("Neural network", "A small dense network mixes standardized features and learns nonlinear combinations. "
            "Its result depends on feature scaling and optimization budget.",
            "Binary cross-entropy with Adam and L2 regularization"),
    "KNeighbors": ("Instance-<br/>based", "Nearby feature vectors can share a road-matching pattern. "
                   "Dense candidate groups and sparse training coverage can make nearest neighbors misleading.",
                   "Distance-weighted neighbor vote, with no fitted loss"),
    "ExtraTrees": ("Randomized trees", "Randomized split thresholds diversify the trees and model interactions "
                  "among distance, heading, and road geometry. Individual splits can miss sharp boundaries.",
                  "Gini impurity at randomized splits"),
}

DISPLAY = {"GaussianNB": "Gaussian NB", "LogisticRegression": "Logistic regression",
           "RandomForest": "Random forest", "HistGradientBoosting": "Histogram GB", "MLP": "MLP",
           "KNeighbors": "K-nearest neighbors", "ExtraTrees": "Extra Trees"}
SUMMARY_CONFIG = {
    "GaussianNB": "Smoothing 1e-9",
    "LogisticRegression": "C 1.0, standardized",
    "RandomForest": "48 trees, depth 13, leaf 8, features 0.8",
    "HistGradientBoosting": "70 iterations, 23 leaves, rate 0.12, L2 0.3",
    "MLP": "24 units, batch 512, rate 0.003, alpha 1e-4",
    "KNeighbors": "k 15, distance weights, 10k fit rows",
    "ExtraTrees": "100 trees, depth 16, leaf 5, bootstrap",
}
REPO = "https://github.com/PES1UG24AM922/MLHackathon-PT119"
MODEL_READOUT = {
    "GaussianNB": "Its low Top-1 in both checks is consistent with a weaker fit than the tree models.",
    "LogisticRegression": "It recovers much of the tree models' accuracy with faster inference, but trails on observation log loss.",
    "RandomForest": "It closes most of the Top-1 and macro F1 gap to boosting and outperforms Extra Trees on both.",
    "HistGradientBoosting": "It has the highest Top-1 and lowest student-test observation log loss among the untuned scorers.",
    "MLP": "Its Top-1 is close to the logistic baseline, with lower observation log loss. The grouped loss curve is on the next model-diagnostics page.",
    "KNeighbors": "It has the highest log loss and slowest scoring in this replay. A zero score on a retrieved reference arc is heavily penalized.",
    "ExtraTrees": "It is less accurate than the random forest here, with fewer false-positive candidate decisions and more false negatives at 0.5.",
}


def pct(value):
    return f"{float(value):.2%}" if pd.notna(value) else "N/A"


def number(value, decimals=3):
    return f"{float(value):.{decimals}f}" if pd.notna(value) else "N/A"


def dataframe(path):
    return pd.read_csv(path, dtype={"trip_id": str})


def make_plots(models, optimization, destinations):
    plt.rcParams.update({"font.size": 10, "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5), layout="constrained")
    y = np.arange(len(models))
    labels = [DISPLAY[model] for model in models.model]
    metrics = [("cv_macro_f1", "Training CV candidate macro F1", True),
               ("student_top1", "Student-test observation Top-1", True),
               ("student_obs_macro_f1", "Student-test observation macro F1", True),
               ("student_obs_log_loss", "Student-test observation log loss", False)]
    for index, (ax, (field, label, is_fraction)) in enumerate(zip(axes.flat, metrics)):
        ax.scatter(models[field], y, color="#bf783c" if not is_fraction else "#007e72", s=25, zorder=3)
        ax.set_yticks(y, labels=labels if index % 2 == 0 else [""] * len(labels))
        ax.set_title(label, fontsize=10)
        if is_fraction:
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.invert_yaxis()
        ax.grid(alpha=.20)
    fig.savefig(destinations / "model_comparison.png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 2.5), layout="constrained")
    ax.scatter(optimization.number + 1, optimization.value, color="#7793ad", label="Trial")
    ax.step(optimization.number + 1, optimization.value.cummax(), where="post", color="#007e72", label="Best so far")
    ax.set(xlabel="Trial", ylabel="Mean CV observation Top-1", xticks=optimization.number + 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=.18)
    fig.savefig(destinations / "bayesian_history.png", dpi=155, bbox_inches="tight")
    plt.close(fig)


def make_report(output, plot_dir, api_screenshot=None):
    eda = dataframe(Path("results/eda_split_summary.csv"))
    misses = dataframe(RESULTS / "replay_candidate_misses.csv")
    diagnostics = dataframe(RESULTS / "student_model_diagnostics.csv")
    models = diagnostics[diagnostics.model.isin(FAMILIES)]
    historical = dataframe(Path("results/model_comparison.csv"))
    folds = dataframe(RESULTS / "diagnostics_folds.csv")
    optimum = json.loads((RESULTS / "model_selection.json").read_text())
    search = dataframe(RESULTS / "bayesian_trials.csv")
    bayes = diagnostics.set_index("model").loc[[optimum["winner"], "Tuned raw", "Tuned sigmoid"]]
    tuning = dataframe(RESULTS / "decoder_trials.csv")
    decoder = dataframe(RESULTS / "classifier_vs_hmm.csv").set_index("method")
    by_trip = dataframe(RESULTS / "classifier_vs_hmm_by_trip.csv")
    api = dataframe(RESULTS / "api_student_test.csv")
    make_plots(models, search, plot_dir)
    grouped_loss = dataframe(RESULTS / "mlp_grouped_loss.csv")
    fig, ax = plt.subplots(figsize=(8.5, 2.7), layout="constrained")
    ax.plot(grouped_loss.epoch, grouped_loss.training_log_loss, color="#007e72", label="Fit trips")
    ax.plot(grouped_loss.epoch, grouped_loss.validation_log_loss, color="#b06928", label="Held-out trips")
    ax.set(xlabel="Epoch", ylabel="Candidate binary log loss")
    ax.grid(alpha=.18)
    ax.legend()
    fig.savefig(plot_dir / "mlp_grouped_loss.png", dpi=155, bbox_inches="tight")
    plt.close(fig)

    styles = getSampleStyleSheet()
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, filename in (
        ("PathSnapSans", "DejaVuSans.ttf"),
        ("PathSnapSans-Bold", "DejaVuSans-Bold.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily("PathSnapSans", normal="PathSnapSans", bold="PathSnapSans-Bold")
    styles["Title"].fontName = styles["Heading2"].fontName = "PathSnapSans-Bold"
    styles["BodyText"].fontName = "PathSnapSans"
    styles["Title"].alignment = TA_LEFT
    styles["Title"].textColor = DARK
    styles["Title"].fontSize = 25
    styles["Title"].leading = 30
    styles["Heading2"].textColor = ACCENT
    styles["Heading2"].spaceAfter = 6
    styles["BodyText"].leading = 14
    styles.add(ParagraphStyle(name="Label", fontName="PathSnapSans-Bold", fontSize=8.7,
                              spaceAfter=8, textColor=ACCENT))
    styles.add(ParagraphStyle(name="Evidence", fontName="PathSnapSans", fontSize=8.2, leading=11,
                              textColor=MUTED, spaceBefore=5, spaceAfter=5))
    styles.add(ParagraphStyle(name="Cell", fontName="PathSnapSans", fontSize=7.6, leading=9.6, textColor=DARK))
    styles.add(ParagraphStyle(name="CellHeader", parent=styles["Cell"], fontName="PathSnapSans-Bold", textColor=colors.white))
    story = []

    def p(body, style="BodyText"):
        story.append(Paragraph(body, styles[style]))

    def title(label, name):
        if story:
            story.append(PageBreak())
        p(label, "Label")
        p(name, "Title")

    def image(path, max_height, width=WIDTH):
        with PixelImage.open(path) as bitmap:
            w, h = bitmap.size
        scale = min(width / w, max_height / h)
        story.append(Image(str(path), width=w * scale, height=h * scale, hAlign="CENTER"))

    def table(rows, widths, wrapped=False):
        values = [[Paragraph(str(value), styles["CellHeader" if index == 0 else "Cell"])
                   for value in row] for index, row in enumerate(rows)] if wrapped else rows
        block = Table(values, colWidths=widths, repeatRows=1)
        block.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "PathSnapSans"),
            ("FONTNAME", (0, 0), (-1, 0), "PathSnapSans-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
        ]))
        story.extend([block, Spacer(1, 8)])

    title("PES UNIVERSITY   /   UE24CS352A   /   TEAM PT119", "PathSnap technical report")
    p("2. Dataset and preprocessing", "Heading2")
    p("PathSnap matches noisy GPS observations to directed road arcs and reconstructs a route. "
      "The development evidence uses 80 training and 10 student-test trips. Reserved evaluator data "
      "was excluded from preprocessing experiments, training, tuning, and this report.")
    rows = [["Partition", "Trips", "GPS rows", "Longest gap", "Missing/invalid trips", "Final candidate miss"]]
    for split in ("train", "student_test"):
        data = eda[eda.split == split].iloc[0]
        candidate = misses[misses.split == split]
        rows.append([split, str(len(candidate)), f"{candidate.n_points.sum():,}", f"{data.dt_max_max:,.0f} s",
                     str(int(data.missing_or_invalid_max)), pct(candidate.n_missed_gt.sum() / candidate.n_points.sum())])
    table(rows, [83, 43, 65, 72, 113, 115])
    p("Raw inputs and reference alignment", "Heading2")
    p("Each raw trip provides GPS longitude, latitude and time, directed road endpoints, and an ordered reference "
      "route. Longitude and latitude are projected to a trip-specific UTM zone for metre-scale operations. "
      "A dynamic program assigns observations to nondecreasing reference occurrences. It minimizes robust "
      "point-to-arc distance with heading and weak step-dependent route-progress costs. Repeated arc IDs remain "
      "distinct during alignment. That alignment supplies training targets and server-side point scoring.")
    p("Candidate generation and features", "Heading2")
    p("We search a spatial index at a 50 m radius, keep the best 20 candidates, and use a nearest-sample fallback "
      "when the radius is empty. Distance, rank and projection position measure proximity. Road direction, speed, "
      "time gaps and distances to neighboring points help distinguish parallel roads, reverse arcs, and GPS gaps. "
      "Candidate misses are reference arcs absent from the resulting search set.")
    image(Path("eda/PathSnap_EDA_outputs/01_gps_observations_hist.png"), 190)
    p("Train and student-test trajectory sizes and durations. Sources: eda/PathSnap_EDA.ipynb, "
      "results/eda_split_summary.csv, and results/final/replay_candidate_misses.csv.", "Evidence")

    title("SECTION 2   /   EXPLORATORY DATA ANALYSIS", "Trajectories and road connectivity")
    image(Path("eda/PathSnap_EDA_outputs/05_reference_vs_gps.png"), 280)
    p("Training trip 00000001 shows GPS drift around parallel roads and junctions. The interactive map "
      "for this corridor is eda/PathSnap_EDA_outputs/06_train_trip_00000001_map.html.")
    image(Path("eda/PathSnap_EDA_outputs/04_connectivity.png"), 225)
    p("Both partitions contain road graphs where weak connectivity exceeds strong connectivity. The decoder uses "
      "directed path checks and starts a new segment after a long gap or an unreachable transition.", "Evidence")

    title("SECTION 3   /   GROUPED VALIDATION", "Model comparison")
    p("Five-fold cross-validation holds out whole training trips. Most fits use up to 80,000 stratified candidate rows. "
      "K-nearest neighbors uses 10,000 to bound search cost. The K-nearest neighbors and Extra Trees ordered-label runs "
      "were added after release model selection. Every model scores the same 19,160 student-test observations.")
    rows = [["Model", "Family", "Initial configuration", "CV<br/>F1", "Test<br/>Top-1", "Test<br/>F1",
             "Test<br/>log loss", "Route<br/>cont.", "Infer<br/>(s)"]]
    for row in models.itertuples():
        rows.append([DISPLAY[row.model], FAMILIES[row.model][0], SUMMARY_CONFIG[row.model], pct(row.cv_macro_f1),
                     pct(row.student_top1), pct(row.student_obs_macro_f1), number(row.student_obs_log_loss),
                     pct(row.student_route_continuity), number(row.student_inference_s)])
    table(rows, [65, 62, 101, 41, 44, 43, 43, 45, 36], wrapped=True)
    p("CV F1 is binary candidate macro F1. Student-test F1 is arc-label macro F1 within each trip, averaged across trips. "
      "Student-test log loss normalizes scores within each GPS observation. An all-zero score group receives a uniform "
      "distribution over its candidates. An absent reference arc has probability zero, clipped to 10<super>-15</super> "
      "for the logarithm. Top-1 and log loss weight each observation equally.")
    p("Continuity is the mean per-trip fraction of adjacent predictions sharing an arc or a directed endpoint, using "
      "positive gaps up to 30 seconds. Inference time covers candidate scoring across all ten student-test trips.")
    p("The replay cache measured an 11.20% training candidate miss rate versus 11.17% in the archived release run. "
      "The original selection, saved model, and decoder logs are retained. Sources: results/final/student_model_diagnostics.csv, "
      "diagnostics_folds.csv, and student_diagnostics_by_trip.csv.", "Evidence")

    title("SECTION 3   /   COMPARISON PLOTS", "Validation and student-test results")
    image(plot_dir / "model_comparison.png", 365)
    original = dataframe(RESULTS / "model_comparison.csv").set_index("model")
    selected = original.loc[optimum["winner"]]
    p(f"Histogram gradient boosting led the original comparison at {pct(selected.cv_top1)} grouped CV Top-1. "
      "Top-1 measures how often the scorer ranks the reference arc first, which is the relevant choice passed "
      "to the sequence decoder.")

    for section, subset in (("A", models.iloc[:4]), ("B", models.iloc[4:])):
        title(f"SECTION 3   /   MODEL NOTES {section}", "Model behavior")
        for row in subset.itertuples():
            _, reasons, loss = FAMILIES[row.model]
            spread = folds[folds.model == row.model].top1.std()
            p(DISPLAY[row.model], "Heading2")
            p(f"{reasons} Training objective: {loss}. Initial setup: {SUMMARY_CONFIG[row.model]}.")
            p(f"Student-test Top-1 {pct(row.student_top1)}, observation macro F1 {pct(row.student_obs_macro_f1)}, "
              f"observation log loss {number(row.student_obs_log_loss)}. Top-1 on ten fit-pool trips was "
              f"{pct(row.fit_pool_top1)}. Grouped CV Top-1 was {pct(row.cv_top1)} (fold SD {pct(spread)}). "
              f"{MODEL_READOUT[row.model]}", "Evidence")
        p("The fit-pool check uses the ten lowest training trip IDs and the full refit. CV uses fold-specific fits "
          "and held-out trips. None showed higher Top-1 in the fit-pool check. Trip difficulty differs across the checks, "
          "so this comparison does not establish how much each model overfits.", "Evidence")

    title("SECTION 3   /   ORIGINAL BASELINES", "Earlier nearest-arc experiment")
    p("The earlier run derived labels from nearest reference arcs. The release uses ordered alignment. "
      "The older results below retain their original label definition and fit settings.")
    historical_names = {"MLPClassifier": "MLP", "RandomForestClassifier": "Random forest",
                        "HistGradientBoosting": "Histogram GB", "LogisticRegression": "Logistic regression",
                        "ExtraTreesClassifier": "Extra Trees", "KNeighborsClassifier": "K-nearest neighbors"}
    historical_settings = {"MLPClassifier": "64 and 32 units, batch 1024, 30 iterations",
                           "RandomForestClassifier": "100 trees, depth 14, leaf 5, up to 200k samples per tree",
                           "HistGradientBoosting": "200 iterations, depth 8, learning rate 0.08",
                           "LogisticRegression": "C 1.0, standardized, up to 500 iterations",
                           "ExtraTreesClassifier": "100 trees, depth 16, leaf 5, bootstrap",
                           "KNeighborsClassifier": "k 15, distance weights, KD tree, 30k fit rows"}
    table([["Model", "Initial configuration", "CV F1", "Test Top-1", "Test F1", "Test loss", "Infer (s)"]] + [
        [historical_names[row.model], historical_settings[row.model], pct(row.cv_macro_f1_mean),
         pct(row.test_top1_acc), pct(row.test_macro_f1), number(row.test_log_loss), number(row.test_infer_time_s)]
        for row in historical.itertuples()
    ], [80, 155, 44, 54, 45, 50, 52], wrapped=True)
    p("F1 and log loss in the original run score binary candidate labels. Sources: results/model_comparison.csv "
      "and src/models/train_compare.py.", "Evidence")

    p("MLP training and validation loss", "Heading2")
    image(plot_dir / "mlp_grouped_loss.png", 177)
    p("This supplementary diagnostic used the same MLP width, batch size, learning rate and regularization as "
      "the model comparison. It used nine fit trips and three held-out trips from the training partition. "
      "Standardization was fitted on the nine fit trips. Candidate log loss was measured after each epoch "
      "for both groups. The diagnostic used a fixed epoch count and did not influence model selection.")
    best_epoch = grouped_loss.loc[grouped_loss.validation_log_loss.idxmin()]
    last_epoch = grouped_loss.iloc[-1]
    p(f"Validation log loss reached {number(best_epoch.validation_log_loss)} at epoch "
      f"{int(best_epoch.epoch)} and ended at {number(last_epoch.validation_log_loss)}. Final fit-trip loss was "
      f"{number(last_epoch.training_log_loss)}. Validation loss did not rise late in this run, so this split "
      "shows no late-epoch overfitting signal.")
    p("Evidence: results/final/mlp_grouped_loss.csv and scripts/mlp_validation_history.py.", "Evidence")

    title("SECTION 4   /   BAYESIAN OPTIMIZATION", "Train-fold search and calibration")
    p(f"Optuna TPE searched {optimum['winner']} for {optimum['trials']} trials, beginning with "
      f"{optimum['startup_trials']} startup trials. The objective was mean held-out Top-1 over five training folds. "
      "Trial and fold scores are retained.")
    table([["Parameter", "Distribution", "Range", "Selected value"]] + [
        [name, mode, f"{lo} to {hi}", number(optimum["best_params"][name], 4)]
        for name, (mode, lo, hi) in optimum["search_space"].items()
    ], [173, 82, 125, 112])
    stage_labels = {optimum["winner"]: "Default", "Tuned raw": "Tuned, raw", "Tuned sigmoid": "Released, calibrated"}
    table([["Stage", "CV Top-1", "CV F1", "Test Top-1", "Test F1", "Test log loss", "Infer (s)"]] + [
        [stage_labels[name], ("N/A: student only" if pd.isna(row.cv_top1) else pct(row.cv_top1)),
         ("N/A: student only" if pd.isna(row.cv_macro_f1) else pct(row.cv_macro_f1)),
         pct(row.student_top1), pct(row.student_obs_macro_f1), number(row.student_obs_log_loss, 4),
         number(row.student_inference_s)]
        for name, row in bayes.iterrows()
    ], [99, 61, 60, 67, 55, 72, 66], wrapped=True)
    image(plot_dir / "bayesian_history.png", 163)
    baseline, tuned = bayes.iloc[:2].itertuples()
    delta = (tuned.cv_top1 - baseline.cv_top1) * 100
    p(f"The ordered-label replay changed mean CV Top-1 by {delta:+.3f} percentage points. "
      f"Student-test Top-1 was {pct(baseline.student_top1)} before tuning and {pct(tuned.student_top1)} after tuning. "
      "The search did not beat the default on its Top-1 objective. The deployed scorer retains the best searched "
      "configuration and sigmoid calibration fitted using out-of-fold training scores.")
    p("The trial plot is from the original run. The paired table replays both raw configurations with identical "
      "folds and student-test trips. The calibrated row runs the saved production artifact on student-test only. "
      "Sources: results/final/bayesian_trials.csv, model_selection.json, and student_model_diagnostics.csv.", "Evidence")

    title("SECTION 5   /   ROUTE RECONSTRUCTION", "Directed HMM and Viterbi")
    p("The hidden state at each GPS observation is a directed candidate arc. Calibrated candidate scores "
      "are normalized within the observation and converted to log-emissions. A transition starts at the "
      "projected position on one arc and ends at the projected position on the next. Its log-score is the "
      "negative absolute difference between shortest directed path length and GPS displacement, divided by "
      "18 m. Speeds above the configured limit receive an additional penalty.")
    p("Viterbi maximizes the sum of emission and transition log-scores. Missing candidates and gaps over 30 "
      "seconds start a new segment. An unreachable arc-to-arc move has a log-score of negative infinity. "
      "If no reachable predecessor remains, the decoder restarts at that observation.")
    p(f"Three decoder configurations were scored on two length-ranked trips per training fold. "
      f"Trial {int(tuning.loc[tuning.mean_cv_top1.idxmax(), 'trial'])} led by mean fold Top-1. "
      "Both student-test methods use the same geometry expansion and ordered reference scoring.")
    table([["Method", "Point Top-1", "Point F1*", "Continuity*", "Ordered route F1*", "Decode ms*"]] + [
        [key, pct(row.point_top1), pct(row.point_macro_f1_trip_mean), pct(row.continuity_trip_mean),
         pct(row.ordered_route_f1_trip_mean), number(row.inference_ms_trip_mean, 1)]
        for key, row in decoder.iterrows()
    ], [74, 74, 65, 73, 108, 86])
    recall_worse = by_trip[by_trip.hmm_ordered_route_recall < by_trip.classifier_ordered_route_recall - 1e-9]
    example = recall_worse.loc[recall_worse.trip_id == "00000076"].iloc[0]
    p(f"HMM improved Top-1 and ordered-route F1 on all {len(by_trip)} student-test trips. Ordered-route recall fell "
      f"on {len(recall_worse)}. For trip {example.trip_id}, recall fell from {pct(example.classifier_ordered_route_recall)} "
      f"to {pct(example.hmm_ordered_route_recall)}, while precision rose from {pct(example.classifier_ordered_route_precision)} "
      f"to {pct(example.hmm_ordered_route_precision)}. The HMM also increased mean decode time from 0.14 s to 10.06 s. "
      "*Mean over trips.")
    table([["Trip", "GPS rows", "Classifier Top-1", "HMM Top-1", "Classifier route F1", "HMM route F1"]] + [
        [row.trip_id, f"{row.n_points:,}", pct(row.classifier_top1), pct(row.hmm_top1),
         pct(row.classifier_ordered_route_f1), pct(row.hmm_ordered_route_f1)]
        for row in by_trip.itertuples()
    ], [68, 55, 89, 77, 103, 88], wrapped=True)
    p("Sources: results/final/classifier_vs_hmm_by_trip.csv and decoder_tuning_folds.csv.", "Evidence")

    title("SECTION 6   /   FOLIUM VISUALIZATION", "Student-test trip 00000000")
    image(RESULTS / "student_test_map.png", 250)
    p("Zoomed view near the first GPS observation. The dashed navy trace is raw GPS, gray shows candidate road arcs, "
      "orange shows the student-test reference, and teal shows the HMM route. The layer controls toggle each overlay.", "Evidence")
    p(f'<a href="{REPO}/blob/main/results/final/student_test_map.html" color="#007e72">Interactive Folium HTML</a>. '
      "Download the file and open it in a browser for the layer controls. Generate it with scripts/final_decode.py.",
      "Evidence")

    title("SECTION 7   /   API, UI AND DEPLOYMENT", "Working UI and API responses")
    image(RESULTS / "demo_route_console.png", 310)
    p("Student-test trip 00000040, with 2,995 GPS points and one predicted segment. The result shows 93.1% point "
      "Top-1 and 90.7% ordered-route F1. Orange marks the GPS trace and teal marks the predicted route.", "Evidence")
    p("Deployment and acceptance checks", "Heading2")
    p("Laptop 1 hosts FastAPI, the calibrated model and the decoder. Laptop 2 opens the route console over the LAN, "
      "selects a trip, and receives the reconstructed geometry and metrics. The server reads the reference only after "
      "prediction, to compute aggregate scores.")
    p(f"The recorded student-test acceptance run returned HTTP 200 for {int((api.status == 200).sum())}/{len(api)} "
      f"trips and downloaded each Folium map. Median server prediction time was {api.duration_ms.median()/1000:.2f} s.")
    p(f'Sources: results/final/api_student_test.csv and '
      f'<a href="{REPO}/blob/main/docs/final-release.md" color="#007e72">the deployment guide</a>.', "Evidence")

    p("HTTP responses on the second laptop", "Heading2")
    if api_screenshot is None:
        image(RESULTS / "demo_second_laptop_log.png", 120)
        p("Team capture from the second laptop. The terminal log records HTTP 200 for three live "
          "POST /predictions requests on student-test trips 00000065, 00000052 and 00000077. "
          "Trip 00000065 also emitted a LARGE_GPS_GAP warning.", "Evidence")
    else:
        image(api_screenshot, 160)
        p("API evidence captured on the second laptop.", "Evidence")
    p("AI assistance was used for implementation support, verification scripts and report editing. The figures and "
      "metrics come from the recorded experiments and live captures.", "Evidence")

    def footer(canvas, doc):
        width, _ = doc.pagesize
        canvas.setStrokeColor(colors.HexColor("#dee7ee"))
        canvas.line(51, 48, width - 51, 48)
        canvas.setFont("PathSnapSans", 8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(51, 33, "PT119  /  PathSnap  /  Student data")
        canvas.drawRightString(width - 51, 33, str(doc.page))

    SimpleDocTemplate(str(output), title="PathSnap Technical Report", author="Team PT119",
                      leftMargin=51, rightMargin=51, topMargin=37, bottomMargin=61).build(
        story, onFirstPage=footer, onLaterPages=footer
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--api-screenshot", type=Path, help="Optional replacement for the second-laptop capture")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        make_report(args.output, Path(temporary), args.api_screenshot)
