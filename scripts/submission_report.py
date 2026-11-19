"""Build the consolidated report from committed evidence and local demo captures."""
import argparse
import json
from ast import literal_eval
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PixelImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

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
                     "Mean class probability from Gini-split trees"),
    "HistGradientBoosting": ("Boosted trees", "Histogram splits capture nonlinear spatial and temporal features "
                             "without enumerating all thresholds. Small leaves and regularization limit complexity.",
                             "Binary log loss over additive trees"),
    "MLP": ("Neural network", "A small dense network mixes standardized features and learns nonlinear combinations. "
            "Its result depends on feature scaling and optimization budget.",
            "Binary cross-entropy with Adam and L2 regularization"),
}


def pct(value):
    return f"{float(value):.2%}" if pd.notna(value) else "N/A"


def number(value, decimals=3):
    return f"{float(value):.{decimals}f}" if pd.notna(value) else "N/A"


def dataframe(path):
    return pd.read_csv(path, dtype={"trip_id": str})


def make_plots(models, optimization, destinations):
    plt.rcParams.update({"font.size": 9, "axes.spines.right": False, "axes.spines.top": False})
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.25), layout="constrained")
    y = np.arange(len(models))
    labels = ["Gaussian NB", "Logistic", "Random Forest", "Histogram GB", "MLP"]
    axes[0].barh(y, models.cv_macro_f1, color="#007e72")
    axes[0].set_yticks(y, labels=labels)
    axes[0].set_xlim(.7, 1)
    axes[0].set_title("CV candidate macro F1")
    axes[1].barh(y - .16, models.student_top1, height=.3, label="Point Top-1", color="#007e72")
    axes[1].barh(y + .16, models.student_macro_f1, height=.3, label="Candidate F1", color="#7793ad")
    axes[1].set_yticks(y, labels=[])
    axes[1].set_xlim(.65, 1)
    axes[1].set_title("Student-test scores")
    axes[1].legend(loc="lower right", fontsize=7)
    axes[2].barh(y, models.student_log_loss, color="#d68a3d")
    axes[2].set_yticks(y, labels=[])
    axes[2].set_title("Student-test log loss")
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.18)
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
    misses = dataframe(RESULTS / "candidate_miss_rate.csv")
    models = dataframe(RESULTS / "model_comparison.csv")
    historical = dataframe(Path("results/model_comparison.csv"))
    folds = dataframe(RESULTS / "model_folds.csv")
    optimum = json.loads((RESULTS / "model_selection.json").read_text())
    search = dataframe(RESULTS / "bayesian_trials.csv")
    bayes = dataframe(RESULTS / "bayesian_comparison.csv")
    tuning = dataframe(RESULTS / "decoder_trials.csv")
    decoder = dataframe(RESULTS / "classifier_vs_hmm.csv").set_index("method")
    by_trip = dataframe(RESULTS / "classifier_vs_hmm_by_trip.csv")
    api = dataframe(RESULTS / "api_student_test.csv")
    make_plots(models, search, plot_dir)
    learning = dataframe(RESULTS / "mlp_learning_history.csv")
    fig, left = plt.subplots(figsize=(8.5, 2.35), layout="constrained")
    right = left.twinx()
    left.plot(learning.iteration, learning.training_loss, color="#007e72")
    right.plot(learning.iteration, learning.internal_validation_accuracy, color="#b06928")
    left.set(xlabel="MLP iteration", ylabel="Training loss")
    right.set_ylabel("Internal validation accuracy", color="#b06928")
    right.spines["right"].set_visible(True)
    fig.savefig(plot_dir / "mlp_learning_history.png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    grouped_loss = dataframe(RESULTS / "mlp_grouped_loss.csv")
    fig, ax = plt.subplots(figsize=(8.5, 2.7), layout="constrained")
    ax.plot(grouped_loss.epoch, grouped_loss.training_log_loss, color="#007e72", label="Fit trips")
    ax.plot(grouped_loss.epoch, grouped_loss.validation_log_loss, color="#b06928", label="Held-out trips")
    ax.set(xlabel="Epoch", ylabel="Candidate binary log loss")
    ax.grid(alpha=.18)
    ax.legend()
    fig.savefig(plot_dir / "mlp_grouped_loss.png", dpi=155, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig)

    styles = getSampleStyleSheet()
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, filename in (
        ("PathSnapSans", "DejaVuSans.ttf"),
        ("PathSnapSans-Bold", "DejaVuSans-Bold.ttf"),
        ("PathSnapMono", "DejaVuSansMono.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily("PathSnapSans", normal="PathSnapSans", bold="PathSnapSans-Bold")
    styles["Title"].fontName = styles["Heading2"].fontName = "PathSnapSans-Bold"
    styles["BodyText"].fontName = "PathSnapSans"
    styles["Code"].fontName = "PathSnapMono"
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

    def table(rows, widths):
        block = Table(rows, colWidths=widths, repeatRows=1)
        block.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), "PathSnapSans"),
            ("FONTNAME", (0, 0), (-1, 0), "PathSnapSans-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PAPER]),
        ]))
        story.extend([block, Spacer(1, 8)])

    title("PATHSNAP   /   TEAM PT119   /   19 NOVEMBER 2026", "Dataset understanding and preprocessing")
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
      "when the radius is empty. The 14 features encode point-to-arc and endpoint distances, relative rank and count, "
      "projection position, length, direction agreement, speed, time gap and neighboring point distances. "
      "The training miss rate includes reference targets absent from the final candidate set, which limits Top-1.")
    image(Path("eda/PathSnap_EDA_outputs/01_gps_observations_hist.png"), 190)
    p("Evidence: results/eda_split_summary.csv and results/final/candidate_miss_rate.csv. The histogram "
      "comes from the executed team EDA notebook.", "Evidence")

    title("SECTION 2   /   EXPLORATORY DATA ANALYSIS", "Trajectories and road connectivity")
    image(Path("eda/PathSnap_EDA_outputs/05_reference_vs_gps.png"), 280)
    p("The trajectory diagnostic compares raw observations with reference geometry. Noise and road geometry make "
      "nearest-arc assignment ambiguous, particularly where arcs run parallel or cross.")
    image(Path("eda/PathSnap_EDA_outputs/04_connectivity.png"), 225)
    p("Network connectivity diagnostic from the team EDA outputs. Directed reachability and long timestamp gaps "
      "are handled by decoder path checks and explicit segment breaks.", "Evidence")

    title("SECTION 3   /   GROUPED VALIDATION", "Five-model comparison")
    p("Each of five folds holds out whole trips. Up to 80,000 stratified candidate rows are used per fit. "
      "Validation and student-test scoring use all candidates. Model selection uses mean training-fold "
      "observation Top-1. The student-test partition is held until model and Bayesian choices are locked.")
    rows = [["Model", "CV F1", "CV Top-1", "Test Top-1", "Test F1", "Log loss", "Continuity", "Infer (s)"]]
    names = ["Gaussian NB", "Logistic", "Random Forest", "Histogram GB", "MLP"]
    for name, row in zip(names, models.itertuples()):
        rows.append([name, pct(row.cv_macro_f1), pct(row.cv_top1), pct(row.student_top1),
                     pct(row.student_macro_f1), number(row.student_log_loss),
                     pct(row.student_route_continuity), number(row.student_inference_s)])
    table(rows, [86, 49, 59, 64, 51, 58, 66, 59])
    image(plot_dir / "model_comparison.png", 195)
    p("Point Top-1 counts observations whose highest scoring arc equals the aligned target. Candidate macro F1 "
      "and log loss evaluate candidate labels. Continuity is the mean trip fraction of adjacent, time-eligible "
      "predictions sharing an arc or a directed endpoint. Scorer time covers all 10 student-test trips.")
    p("Evidence: model_folds.csv, model_comparison.csv and model_continuity_by_trip.csv under results/final.", "Evidence")

    for section, subset in (("A", models.iloc[:3]), ("B", models.iloc[3:])):
        title(f"SECTION 3   /   MODEL NOTES {section}", "Algorithms and observed results")
        for row in subset.itertuples():
            family, reasons, loss = FAMILIES[row.model]
            spread = folds[folds.model == row.model].top1.std()
            p(f"{row.model} · {family}", "Heading2")
            config = ", ".join(f"{key}={value}" for key, value in json.loads(row.configuration).items())
            if row.model == "MLP":
                config += ", batch 512, maximum 55 iterations, early stopping"
            p(f"{reasons} Objective: {loss}. Settings: {config}.")
            p(f"Student Top-1 {pct(row.student_top1)}, macro F1 {pct(row.student_macro_f1)}, log loss "
              f"{number(row.student_log_loss)}. CV Top-1 {pct(row.cv_top1)}, with fold SD {pct(spread)}. "
              "The different trip partitions affect the observed gap. Without a separate training-score comparison, "
              "the figures alone do not establish overfitting.")
        if section == "B":
            image(plot_dir / "mlp_learning_history.png", 138)
            p("Actual MLP training loss and internal early-stopping validation accuracy. That internal "
              "candidate split is distinct from the external trip-grouped evaluation.", "Evidence")
            p("Historical experiments before ordered labels", "Heading2")
            p("The earlier team comparison also included K-nearest neighbors and Extra Trees. The estimates below "
              "use the old alignment and are separate from the new model selection.")
            for key, description in [
                ("KNeighborsClassifier", "Distance-weighted k=15 on 30,000 training rows uses local similarity "
                 "and has no optimized training loss. Neighbor searches raised inference cost."),
                ("ExtraTreesClassifier", "100 trees with depth 16 and a 200,000-row cap used randomized split "
                 "thresholds and Gini impurity. Randomization trades split precision for diversity.")
            ]:
                row = historical[historical.model == key].iloc[0]
                p(f"<b>{key}</b>: {description} Top-1 {pct(row.test_top1_acc)}, macro F1 {pct(row.test_macro_f1)}, "
                  f"log loss {number(row.test_log_loss)}. The older run does not isolate generalization from label changes.")

    title("SECTION 3   /   LEARNING CURVES", "MLP training and validation loss")
    image(plot_dir / "mlp_grouped_loss.png", 210)
    p("This supplementary diagnostic used the same MLP width, batch size, learning rate and regularization as "
      "the model comparison. It used nine fit trips and three held-out trips from the training partition. "
      "Standardization was fitted on the nine fit trips. Candidate log loss was measured after each epoch "
      "for both groups. The diagnostic used a fixed epoch count and did not influence model selection.")
    best_epoch = grouped_loss.loc[grouped_loss.validation_log_loss.idxmin()]
    last_epoch = grouped_loss.iloc[-1]
    p(f"Validation log loss reached {number(best_epoch.validation_log_loss)} at epoch "
      f"{int(best_epoch.epoch)} and ended at {number(last_epoch.validation_log_loss)}. Final fit-trip loss was "
      f"{number(last_epoch.training_log_loss)}. These curves describe the recorded trip split.")
    p("Evidence: results/final/mlp_grouped_loss.csv and scripts/mlp_validation_history.py.", "Evidence")

    title("SECTION 4   /   BAYESIAN OPTIMIZATION", "Train-fold search and calibration")
    p(f"Optuna TPE searched {optimum['winner']} for {optimum['trials']} trials, beginning with "
      f"{optimum['startup_trials']} startup trials. The objective was mean held-out Top-1 over five training folds. "
      "Trial and fold scores are retained.")
    table([["Parameter", "Distribution", "Range", "Selected value"]] + [
        [name, mode, f"{lo} to {hi}", number(optimum["best_params"][name], 4)]
        for name, (mode, lo, hi) in optimum["search_space"].items()
    ], [173, 82, 125, 112])
    table([["Stage", "CV Top-1", "CV F1", "Test Top-1", "Test F1", "Test log loss", "Infer (s)"]] + [
        [row.stage, pct(row.cv_top1), pct(row.cv_macro_f1), pct(row.student_top1),
         pct(row.student_macro_f1), number(row.student_log_loss), number(row.student_inference_s)]
        for row in bayes.itertuples()
    ], [98, 65, 58, 68, 57, 78, 68])
    image(plot_dir / "bayesian_history.png", 181)
    diff = (bayes.iloc[1].cv_top1 - bayes.iloc[0].cv_top1) * 100
    p(f"The best searched trial changed mean CV Top-1 by {diff:+.3f} percentage points versus the default. "
      "The release uses that searched configuration with sigmoid calibration fitted from training out-of-fold "
      "scores. Separately, outer-fold decoder probabilities use calibrators that excluded those outer trips. "
      "No calibrated CV estimate is shown for the refitted production scorer.")

    title("SECTION 5   /   ROUTE RECONSTRUCTION", "Directed HMM and Viterbi")
    p("Candidate directed arcs are hidden states. The decoder normalizes calibrated candidate probabilities per "
      "observation and uses their log scores. Transition scores compare projected GPS displacement with directed "
      "network distance including offsets on both arcs. Implausible speeds are penalized. Absent candidates, "
      "time gaps and unreachable transitions split the sequence.")
    p("Decoder parameter selection", "Heading2")
    p(f"Three configurations were evaluated on two predetermined length-ranked trips per training fold. "
      f"Trial {int(tuning.loc[tuning.mean_cv_top1.idxmax(), 'trial'])} won by mean fold Top-1. The student-test comparison "
      "uses all ten trips. Both classifier-only and HMM paths use the same geometry expansion for ordered-route scoring.")
    table([["Method", "Point Top-1", "Point F1*", "Continuity*", "Ordered route F1*", "Decode ms*"]] + [
        [key, pct(row.point_top1), pct(row.point_macro_f1_trip_mean), pct(row.continuity_trip_mean),
         pct(row.ordered_route_f1_trip_mean), number(row.inference_ms_trip_mean, 1)]
        for key, row in decoder.iterrows()
    ], [78, 77, 65, 74, 107, 91])
    worse = by_trip[by_trip.hmm_top1 < by_trip.classifier_top1 - 1e-9]
    p(f"HMM Top-1 changed from {pct(decoder.loc['classifier', 'point_top1'])} to "
      f"{pct(decoder.loc['hmm', 'point_top1'])}. HMM Top-1 was lower on {len(worse)} of "
      f"{len(by_trip)} trips. Continuity and ordered overlap score different aspects of route quality, "
      "so an improvement on one does not guarantee improvements on the others. *Mean over trips.")
    table([["Trip", "GPS rows", "Classifier Top-1", "HMM Top-1", "HMM ordered route F1"]] + [
        [row.trip_id, f"{row.n_points:,}", pct(row.classifier_top1), pct(row.hmm_top1),
         pct(row.hmm_ordered_route_f1)] for row in by_trip.itertuples()
    ], [85, 65, 121, 94, 127])

    title("SECTION 6   /   FOLIUM VISUALIZATION", "Student-test trip 00000000")
    image(RESULTS / "student_test_map.png", 450)
    p("The Folium export shows raw GPS, the candidate road network, the reference and the HMM route. "
      "The file includes layer controls and runs offline. This reference overlay is for student-test analysis. "
      "Official evaluator labels and routes never appear in the demo UI or its exports.")
    p("Interactive evidence: results/final/student_test_map.html. Recreate it with scripts/final_decode.py.", "Evidence")

    title("SECTION 7   /   API, UI AND DEPLOYMENT", "Live student-test verification")
    image(RESULTS / "student_test_ui.png", 345)
    p("Current student-test UI. The orange dashed GPS trace and teal predicted route can be toggled independently.",
      "Evidence")
    p(f"{int((api.status == 200).sum())}/{len(api)} student-test requests returned HTTP 200 and downloadable "
      f"Folium maps. Median prediction latency was {api.duration_ms.median()/1000:.2f} s. Predictions were "
      "made without opening the reference. After prediction, only aggregate metrics are returned.")
    receipt = api.iloc[0]
    p(f"Earlier acceptance-run response: HTTP {receipt.status}, trip {receipt.trip_id}, request {receipt.request_id}, "
      f"{receipt.gps_points} points, point Top-1 {pct(receipt.point_top1_accuracy)}.", "Evidence")
    if api_screenshot is not None:
        p("Receiving laptop terminal", "Heading2")
        image(api_screenshot, 155)
    else:
        p("Recorded API acceptance output", "Heading2")
        log = (RESULTS / "api_verification.log").read_text(encoding="utf-8")
        entry = literal_eval(log.splitlines()[0])
        fields = ("trip_id", "status", "request_id", "gps_points", "check")
        excerpt = "\n".join(f"{key}: {entry[key]}" for key in fields)
        story.append(Preformatted(excerpt, styles["Code"]))
        p("Selected fields from results/final/api_verification.log.", "Evidence")
    p("Laptop 1 runs the model and API. Laptop 2 opens the UI over the LAN. See docs/final-release.md. "
      "This is local verification evidence. Capture the receiving laptop terminal and record all ten official "
      "evaluator choices on the two physical machines during the authorized run.")
    p("Provenance: team source and experimental artifacts are recorded in Git. AI assisted the independent "
      "implementation changes, verification scripts, and report drafting. All reported experiments refer to saved runs.",
      "Evidence")

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
    parser.add_argument("--api-screenshot", type=Path, help="Screenshot captured on the receiving laptop")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        make_report(args.output, Path(temporary), args.api_screenshot)
