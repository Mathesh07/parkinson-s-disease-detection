"""
Gait Parkinson's Disease Detection - Complete Visualization Generator
Generates all presentation-ready figures from existing experiment outputs.
No data is fabricated. All numbers come from saved CSV/JSON files.
Usage: python Gait/visualizations/generate_all_visualizations.py
"""
import sys, warnings
from pathlib import Path

_THIS     = Path(__file__).resolve()
_VIZ_DIR  = _THIS.parent
_GAIT_DIR = _VIZ_DIR.parent
_ROOT_DIR = _GAIT_DIR.parent
for _p in (str(_GAIT_DIR), str(_ROOT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
S = {
    "dpi": 300, "bg": "#0f1117", "panel": "#1a1d27", "grid": "#2a2d3a",
    "text": "#e8e8f0",
    "a1": "#4f8ef7", "a2": "#f7614f", "a3": "#4fcf8e",
    "a4": "#f7c24f", "a5": "#c44ff7",
    "cohort": {"Si": "#4f8ef7", "Ju": "#4fcf8e", "Ga": "#f7614f"},
    "method": {
        "Softmax": "#6b93d6", "Temperature Scaling": "#4fcf8e",
        "EDL": "#f7c24f", "MC Dropout": "#c44ff7", "Deep Ensemble": "#f7614f",
    },
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.facecolor": S["panel"], "figure.facecolor": S["bg"],
    "axes.edgecolor": S["grid"], "axes.labelcolor": S["text"],
    "xtick.color": S["text"], "ytick.color": S["text"], "text.color": S["text"],
    "grid.color": S["grid"], "grid.linestyle": "--", "grid.alpha": 0.4,
    "legend.framealpha": 0.2, "legend.edgecolor": S["grid"],
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_BASE = _GAIT_DIR / "outputs"
DATA_DIR = _GAIT_DIR / "gait-in-parkinsons-disease-1.0.0"
PATHS = {
    "loco":      OUT_BASE / "loco"             / "loco_results.csv",
    "loco_subj": OUT_BASE / "loco"             / "loco_subject_predictions.csv",
    "mseed":     OUT_BASE / "multiseed"        / "multiseed_summary.csv",
    "cal":       OUT_BASE / "calibration"      / "calibration_results.csv",
    "cal_subj":  OUT_BASE / "calibration"      / "calibration_subject_predictions.csv",
    "ens":       OUT_BASE / "deep_ensemble"    / "deep_ensemble_results.csv",
    "ens_cmp":   OUT_BASE / "deep_ensemble"    / "deep_ensemble_comparison.csv",
    "conf":      OUT_BASE / "conformal"        / "conformal_results.csv",
    "conf_corr": OUT_BASE / "conformal"        / "conformal_corruption_coverage.csv",
    "corr":      OUT_BASE / "corruption_shift" / "corruption_shift_results.csv",
    "rel":       OUT_BASE / "reliability"      / "reliability_results.csv",
    "rel_subj":  OUT_BASE / "reliability"      / "reliability_subject_decisions.csv",
}
OUT = {
    "arch":   _VIZ_DIR / "architecture",
    "pipe":   _VIZ_DIR / "data_pipeline",
    "perf":   _VIZ_DIR / "performance",
    "cal":    _VIZ_DIR / "calibration",
    "unc":    _VIZ_DIR / "uncertainty",
    "cohort": _VIZ_DIR / "cohort_shift",
    "conf":   _VIZ_DIR / "conformal",
    "rel":    _VIZ_DIR / "reliability",
    "inf":    _VIZ_DIR / "inference",
}
for _d in OUT.values():
    _d.mkdir(parents=True, exist_ok=True)

GEN, SKIP = [], []
_SVG_KEYS = ("architecture", "pipeline", "flow", "protocol",
             "explained", "story", "overview", "inference")

def _save(fig, path):
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=S["dpi"], bbox_inches="tight", facecolor=fig.get_facecolor())
    GEN.append(str(png))
    if any(k in str(path) for k in _SVG_KEYS):
        svg = path.with_suffix(".svg")
        fig.savefig(svg, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
        GEN.append(str(svg))
    plt.close(fig)
    try:
        print(f"  OK  {png.relative_to(_ROOT_DIR)}")
    except Exception:
        print(f"  OK  {png}")

def _skip(label, reason):
    SKIP.append((label, reason))
    print(f"  --  SKIP {label}: {reason}")

def _load(key):
    p = PATHS.get(key)
    return pd.read_csv(p) if (p and p.exists()) else None

# ===========================================================================
# 1. MODEL ARCHITECTURE DIAGRAM
# ===========================================================================
def fig_model_architecture():
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13); ax.axis("off")
    layers = [
        ("16-channel VGRF Input",                        12.2, "#2d6a9f",
         "(B, 500, 16)   100 Hz   5-second window"),
        ("Transpose  (B, 16, 500)",                      11.2, "#1e4d72",
         "Prepare for Conv1D"),
        ("CNN Block 1  Conv1D + BN + ReLU + MaxPool",     9.9, "#1a7a5e",
         "16->64 channels   kernel=5   500->250 samples"),
        ("CNN Block 2  Conv1D + BN + ReLU + MaxPool",     8.5, "#1a7a5e",
         "64->128 channels   kernel=5   250->125 samples"),
        ("2-Layer Bidirectional LSTM",                    7.0, "#6b3fa0",
         "hidden=128   bidir   output=(B, 125, 256)"),
        ("Global Temporal Mean Pooling",                  5.9, "#4a3070",
         "Mean across 125 time steps  ->  (B, 256)"),
        ("Embedding  Linear + BN + ReLU + Dropout",       4.6, "#8b6914",
         "256  ->  128-dimensional Gait Feature Vector"),
        ("Evidential Head  Dropout + Linear + Softplus",  3.2, "#7a2020",
         "128  ->  [e_HC, e_PD]   evidence >= 0"),
        ("PD / Healthy Prediction",                       1.8, "#1a5e2d",
         "Calibrated via Temperature Scaling"),
    ]
    bx, bw, bh = 1.5, 7.0, 0.72
    for i, (lbl, yc, col, sub) in enumerate(layers):
        ax.add_patch(FancyBboxPatch((bx, yc - bh/2), bw, bh,
            boxstyle="round,pad=0.08", linewidth=1.2,
            edgecolor="#ffffff44", facecolor=col + "cc"))
        ax.text(bx + bw/2, yc + 0.09, lbl, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white")
        ax.text(bx + bw/2, yc - 0.27, sub, ha="center", va="center",
                fontsize=7.8, color="#ccccdd", style="italic")
        if i < len(layers) - 1:
            nyc = layers[i + 1][1]
            ax.annotate("", xy=(5.0, nyc + bh/2 + 0.03),
                        xytext=(5.0, yc - bh/2 - 0.03),
                        arrowprops=dict(arrowstyle="->", color="#aaaacc", lw=1.5))
    for txt, yc, col in [("LOCAL\nFEATURES", 9.2, "#1a7a5e"),
                          ("SEQUENCE\nMODELLING", 6.5, "#6b3fa0"),
                          ("UNCERTAINTY\nOUTPUT", 3.0, "#7a2020")]:
        ax.text(9.6, yc, txt, ha="center", va="center", fontsize=7.5,
                color=col, fontweight="bold", linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col + "22",
                          edgecolor=col + "88", lw=1))
    ax.set_title("GaitCNNBiLSTM Architecture\n1D CNN + 2-Layer BiLSTM + Evidential Head",
                 fontsize=14, fontweight="bold", pad=10)
    _save(fig, OUT["arch"] / "gait_model_architecture")


# ===========================================================================
# 2. END-TO-END PIPELINE
# ===========================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 12); ax.set_ylim(0, 16); ax.axis("off")
    phases = [
        ("Raw PhysioNet Gait Recording (.txt)",  15.3, "#1e3a5f", "DATA",
         "PhysioNet  3 cohorts: Ga / Ju / Si"),
        ("16 VGRF Channels",                     14.1, "#1e3a5f", "DATA",
         "Left L1-L8   Right R1-R8   100 Hz"),
        ("Training-Only Normalization",           12.9, "#1a5740", "TRAIN",
         "Channel-wise z-score   fit on training data only"),
        ("Sliding Window Segmentation",           11.7, "#1a5740", "PREPROC",
         "5 sec / 500 samples   50% overlap / 250-step shift"),
        ("1D CNN Feature Extraction",             10.4, "#3d1f72", "MODEL",
         "Block 1: 16->64   Block 2: 64->128   MaxPool x2"),
        ("2-Layer BiLSTM",                         9.1, "#3d1f72", "MODEL",
         "hidden=128  bidirectional  global mean pooling"),
        ("128-D Gait Embedding",                   7.9, "#3d1f72", "MODEL",
         "Compact gait feature vector"),
        ("Evidential Head -> PD Probability",      6.7, "#3d1f72", "MODEL",
         "Softplus evidence   [e_HC, e_PD] >= 0"),
        ("Temperature Scaling (Calibration)",      5.5, "#725a1a", "CALIB",
         "Learned T per LOCO fold   improves NLL and ECE"),
        ("Uncertainty Estimation",                 4.3, "#5a1a72", "UNC",
         "Confidence=max(P_HC,P_PD)   Uncertainty=1-Conf"),
        ("Conformal Prediction Set",               3.1, "#5a1a72", "CONF",
         "alpha=0.10   {HC}, {PD}, or {HC,PD}"),
        ("Reliability Layer  ->  ACCEPT / FLAG",   1.8, "#1a5a2d", "DEC",
         "tau_conf  tau_unc  q_hat per fold"),
    ]
    bx, bw, bh = 2.2, 7.5, 0.70
    for i, (lbl, yc, col, phase, sub) in enumerate(phases):
        ax.add_patch(FancyBboxPatch((bx, yc - bh/2), bw, bh,
            boxstyle="round,pad=0.07", linewidth=1.0,
            edgecolor="#ffffff33", facecolor=col + "dd"))
        ax.text(bx + 0.2, yc, phase, ha="left", va="center",
                fontsize=6.5, fontweight="bold", color="#ffffffaa",
                bbox=dict(boxstyle="round,pad=0.18", facecolor=col,
                          edgecolor="none", alpha=0.6))
        ax.text(bx + bw/2 + 0.3, yc + 0.12, lbl, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(bx + bw/2 + 0.3, yc - 0.24, sub, ha="center", va="center",
                fontsize=7.5, color="#ccccdd", style="italic")
        if i < len(phases) - 1:
            nyc = phases[i + 1][1]
            ax.annotate("", xy=(5.95, nyc + bh/2 + 0.04),
                        xytext=(5.95, yc - bh/2 - 0.04),
                        arrowprops=dict(arrowstyle="->", color="#9999bb", lw=1.5))
    for xp, lbl, col in [(4.5, "ACCEPT\nReliable Prediction", "#1a7a3a"),
                          (8.2, "FLAG\nAbstain / Review",      "#7a2020")]:
        ax.add_patch(FancyBboxPatch((xp - 1.2, 0.25), 2.5, 0.95,
            boxstyle="round,pad=0.10", linewidth=1.5,
            edgecolor=col, facecolor=col + "44"))
        ax.text(xp + 0.05, 0.72, lbl, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=col, linespacing=1.4)
    ax.annotate("", xy=(4.55, 1.2), xytext=(5.0, 1.45),
                arrowprops=dict(arrowstyle="->", color="#1a7a3a", lw=2.0))
    ax.annotate("", xy=(8.25, 1.2), xytext=(7.5, 1.45),
                arrowprops=dict(arrowstyle="->", color="#7a2020", lw=2.0))
    ax.set_title("End-to-End Gait Parkinson's Detection Pipeline\nFrom Raw Signal to Reliability-Aware Prediction",
                 fontsize=13, fontweight="bold", pad=8)
    _save(fig, OUT["pipe"] / "end_to_end_pipeline")

# ===========================================================================
# 3. REAL GAIT SIGNAL & WINDOWING
# ===========================================================================
def _load_raw_gait():
    for c in [DATA_DIR / "GaCo01_01.txt", DATA_DIR / "SiCo01_01.txt",
               DATA_DIR / "JuCo01_01.txt"]:
        if c.exists():
            try:
                arr = np.loadtxt(c)
                if arr.ndim == 2 and arr.shape[1] >= 17:
                    return arr[:1500, 1:17], c.name
            except Exception:
                pass
    for txt in DATA_DIR.glob("*.txt"):
        try:
            arr = np.loadtxt(txt)
            if arr.ndim == 2 and arr.shape[1] >= 17:
                return arr[:1500, 1:17], txt.name
        except Exception:
            pass
    return None, ""

def fig_gait_signal():
    raw, fname = _load_raw_gait()
    if raw is None:
        _skip("example_gait_signal.png", "No readable dataset .txt file found"); return
    t = np.arange(len(raw)) / 100.0
    names = ["L1","L2","L3","L4","L5","L6","L7","L8",
             "R1","R2","R3","R4","R5","R6","R7","R8"]
    fig, axes = plt.subplots(16, 1, figsize=(14, 14), sharex=True)
    fig.patch.set_facecolor(S["bg"])
    for ch, ax in enumerate(axes):
        col = S["a1"] if ch < 8 else S["a2"]
        ax.plot(t, raw[:, ch], lw=0.7, color=col, alpha=0.9)
        ax.set_ylabel(names[ch], fontsize=7, rotation=0, labelpad=22,
                      va="center", color=col)
        ax.set_yticks([])
        ax.set_facecolor(S["panel"])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    fig.legend(handles=[mpatches.Patch(color=S["a1"], label="Left Foot L1-L8"),
                        mpatches.Patch(color=S["a2"], label="Right Foot R1-R8")],
               loc="upper right", fontsize=9, framealpha=0.15)
    fig.suptitle(f"Real VGRF Gait Recording - {fname}\n"
                 "16-Channel Ground Reaction Force (15-second excerpt, 100 Hz)",
                 fontsize=12, fontweight="bold", y=0.995)
    plt.subplots_adjust(hspace=0.04, top=0.96)
    _save(fig, OUT["pipe"] / "example_gait_signal")

def fig_windowing():
    raw, _ = _load_raw_gait()
    if raw is None:
        _skip("windowing_example.png", "No readable dataset file"); return
    sig = raw.sum(axis=1)
    t   = np.arange(len(sig)) / 100.0
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t, sig, color=S["a1"], lw=0.9, alpha=0.85, label="Summed VGRF (16 channels)")
    for i, (start, col) in enumerate(zip([100, 350, 600], [S["a3"], S["a4"], S["a5"]])):
        ts, te = start / 100, (start + 500) / 100
        ax.axvspan(ts, te, alpha=0.20, color=col)
        ax.axvline(ts, color=col, lw=1.4, ls="--", alpha=0.7)
        ax.axvline(te, color=col, lw=1.4, ls="--", alpha=0.7)
        ymax = sig[start:start + 500].max()
        ax.text((ts + te) / 2, ymax * 1.03, f"Window {i+1}\n5 sec / 500 samples",
                ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")
    ov_s = 350 / 100; ov_e = (100 + 500) / 100; mid = (ov_s + ov_e) / 2
    yb = sig[100 + 250] * 0.5
    ax.annotate("", xy=(ov_e, yb), xytext=(ov_s, yb),
                arrowprops=dict(arrowstyle="<->", color=S["a2"], lw=2.0))
    ax.text(mid, yb * 1.15, "50% overlap\n250 samples / 2.5 sec",
            ha="center", va="bottom", fontsize=8.5, color=S["a2"], fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("VGRF Magnitude (N)", fontsize=11)
    ax.set_title("Sliding Window Segmentation - 5-Second Windows with 50% Overlap\n"
                 "100 Hz   500 samples per window   250-sample step",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    _save(fig, OUT["pipe"] / "windowing_example")


# ===========================================================================
# 4. LOCO PROTOCOL
# ===========================================================================
def fig_loco_protocol():
    df = _load("loco")
    if df is None:
        _skip("loco_protocol.png", "loco_results.csv not found"); return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.5); ax.axis("off")
    df3 = df[df["experiment"].isin([1, 2, 3])].reset_index(drop=True)
    cxs = [2.0, 6.0, 10.0]
    for i, row in df3.iterrows():
        cx    = cxs[i]
        test  = str(row["test_cohort"])
        train = str(row["train_cohorts"]).split("+")
        acc   = float(row["accuracy"]) * 100
        auc   = float(row["roc_auc"])
        col   = S["cohort"].get(test, S["a1"])
        ax.add_patch(FancyBboxPatch((cx - 1.4, 5.4), 2.8, 0.8,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor=col, facecolor=col + "33"))
        ax.text(cx, 5.8, f"Experiment {int(row['experiment'])}",
                ha="center", va="center", fontsize=11, fontweight="bold", color=col)
        for j, tc in enumerate(train):
            ty = 4.3 - j * 1.0
            ax.add_patch(FancyBboxPatch((cx - 1.2, ty - 0.32), 2.4, 0.64,
                boxstyle="round,pad=0.08", linewidth=1,
                edgecolor="#4fcf8e99", facecolor="#1a5740cc"))
            ax.text(cx, ty, f"Train: {tc}", ha="center", va="center",
                    fontsize=9.5, color=S["a3"], fontweight="bold")
        ax.annotate("", xy=(cx, 2.55), xytext=(cx, 3.15),
                    arrowprops=dict(arrowstyle="->", color="#aaaacc", lw=1.8))
        ax.add_patch(FancyBboxPatch((cx - 1.2, 1.7), 2.4, 0.82,
            boxstyle="round,pad=0.08", linewidth=2,
            edgecolor=col, facecolor=col + "44"))
        ax.text(cx, 2.11, f"TEST: {test}", ha="center", va="center",
                fontsize=10, color=col, fontweight="bold")
        ax.text(cx, 1.53, f"Acc={acc:.1f}%  AUC={auc:.2f}",
                ha="center", va="center", fontsize=8.5, color="#ddddee")
    ax.text(6.0, 0.55,
            "Test cohort is completely unseen during training and validation",
            ha="center", va="center", fontsize=10, color=S["a4"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#3a3010",
                      edgecolor=S["a4"] + "88", lw=1))
    ax.set_title("Leave-One-Cohort-Out (LOCO) Evaluation Protocol\n"
                 "Each cohort serves as the unseen test set once",
                 fontsize=14, fontweight="bold", pad=8)
    _save(fig, OUT["cohort"] / "loco_protocol")


# ===========================================================================
# 5. LOCO PERFORMANCE
# ===========================================================================
def _loco_bar(metric, ylabel, title, fname, pct=False):
    df = _load("loco")
    if df is None:
        _skip(fname, "loco_results.csv not found"); return
    df = df[df["experiment"].isin([1, 2, 3])].reset_index(drop=True)
    if metric not in df.columns:
        _skip(fname, f"Column {metric} not found"); return
    cohorts = list(df["test_cohort"].astype(str))
    vals    = [float(v) * (100 if pct else 1) for v in df[metric]]
    colors  = [S["cohort"].get(c, S["a1"]) for c in cohorts]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(cohorts, vals, color=colors, width=0.5, alpha=0.88, edgecolor="#ffffff22")
    for bar, val in zip(bars, vals):
        lbl = f"{val:.1f}%" if pct else f"{val:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.4 if pct else 0.005),
                lbl, ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Test Cohort (LOCO Fold)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.35)
    mx = max(vals) if vals else 1
    ax.set_ylim(0, mx * 1.18 if mx > 0 else 1)
    _save(fig, OUT["perf"] / fname)

def fig_loco_performance():
    _loco_bar("accuracy",          "Accuracy (%)",          "LOCO Accuracy by Test Cohort",          "loco_accuracy",          pct=True)
    _loco_bar("balanced_accuracy", "Balanced Accuracy (%)", "LOCO Balanced Accuracy by Test Cohort", "loco_balanced_accuracy", pct=True)
    _loco_bar("roc_auc",           "ROC-AUC",               "LOCO ROC-AUC by Test Cohort",           "loco_auc")
    _loco_bar("nll",               "Neg Log-Likelihood",    "LOCO NLL by Test Cohort",               "loco_nll")
    _loco_bar("brier",             "Brier Score",           "LOCO Brier Score by Test Cohort",       "loco_brier")

# ===========================================================================
# 6. MULTI-SEED
# ===========================================================================
def fig_multiseed_performance():
    df = _load("mseed")
    if df is None:
        _skip("multiseed_performance.png", "multiseed_summary.csv not found"); return
    metrics = ["accuracy", "balanced_accuracy", "roc_auc", "f1"]
    labels  = ["Accuracy", "Balanced Accuracy", "ROC-AUC", "F1 Score"]
    groups  = [("Exp_1_Si", "Si"), ("Exp_2_Ju", "Ju"), ("Exp_3_Ga", "Ga")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, metric, lbl in zip(axes, metrics, labels):
        for grp, cohort in groups:
            row = df[(df["group"] == grp) & (df["metric"] == metric)]
            if row.empty: continue
            mean = float(row["mean"].iloc[0]) * 100
            lo   = float(row["ci95_lower"].iloc[0]) * 100
            hi   = float(row["ci95_upper"].iloc[0]) * 100
            col  = S["cohort"].get(cohort, S["a1"])
            ax.bar(cohort, mean, color=col, width=0.5, alpha=0.85, edgecolor="#ffffff22")
            ax.errorbar(cohort, mean, yerr=[[mean - lo], [hi - mean]],
                        fmt="none", color="white", capsize=5, lw=2)
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.set_ylabel("Score (%)", fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Multi-Seed LOCO Performance (Seeds 42-46)   Mean +/- 95% CI",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, OUT["perf"] / "multiseed_performance")

def fig_multiseed_uncertainty():
    df = _load("mseed")
    if df is None:
        _skip("multiseed_uncertainty.png", "multiseed_summary.csv not found"); return
    groups = [("Exp_1_Si", "Si"), ("Exp_2_Ju", "Ju"), ("Exp_3_Ga", "Ga")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (grp, cohort) in zip(axes, groups):
        col = S["cohort"].get(cohort, S["a1"])
        for metric, lbl, pat in [
            ("mean_uncertainty_correct",   "Correct",   "/"),
            ("mean_uncertainty_incorrect", "Incorrect", "\\"),
        ]:
            row = df[(df["group"] == grp) & (df["metric"] == metric)]
            if row.empty: continue
            mean = float(row["mean"].iloc[0])
            lo   = float(row["ci95_lower"].iloc[0])
            hi   = float(row["ci95_upper"].iloc[0])
            ax.bar(lbl, mean, hatch=pat, edgecolor=col,
                   facecolor=col + "66", width=0.4, lw=1.5)
            ax.errorbar(lbl, mean, yerr=[[mean - lo], [hi - mean]],
                        fmt="none", color="white", capsize=5, lw=2)
        ax.set_title(f"Test Cohort: {cohort}", fontsize=11,
                     fontweight="bold", color=col)
        ax.set_ylabel("Mean Uncertainty (1-Confidence)", fontsize=9)
        ax.set_ylim(0, 0.7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Multi-Seed Uncertainty: Correct vs Incorrect   Mean +/- 95% CI",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, OUT["perf"] / "multiseed_uncertainty")


# ===========================================================================
# 7. DEEP ENSEMBLE
# ===========================================================================
def fig_ensemble_schematic():
    df = _load("ens")
    if df is None:
        _skip("deep_ensemble_prediction.png", "deep_ensemble_results.csv not found"); return
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis("off")
    ys  = [6.2, 5.2, 4.2, 3.2, 2.2]
    cls = [S["a1"], S["a3"], S["a4"], S["a5"], S["a2"]]
    for i, (y, col) in enumerate(zip(ys, cls)):
        ax.add_patch(FancyBboxPatch((0.3, y - 0.32), 2.8, 0.64,
            boxstyle="round,pad=0.08", edgecolor=col + "88", facecolor=col + "33"))
        ax.text(1.7, y, f"Model {i+1}  (seed={42+i})",
                ha="center", va="center", fontsize=9.5, color=col, fontweight="bold")
        ax.annotate("", xy=(5.4, 4.2), xytext=(3.1, y),
                    arrowprops=dict(arrowstyle="->", color=col + "88", lw=1.3))
    ax.add_patch(FancyBboxPatch((5.3, 3.5), 2.9, 1.4,
        boxstyle="round,pad=0.1", linewidth=2,
        edgecolor=S["a2"], facecolor=S["a2"] + "33"))
    ax.text(6.75, 4.4, "Ensemble", ha="center", va="center",
            fontsize=11, fontweight="bold", color=S["a2"])
    ax.text(6.75, 4.0, "Mean Prediction\n+ Disagreement Score",
            ha="center", va="center", fontsize=8.5, color="#ccccdd", linespacing=1.4)
    for yd, lbl, col in [(5.8, "High Agreement -> Low Uncertainty",    S["a3"]),
                          (2.5, "High Disagreement -> High Uncertainty", S["a2"])]:
        ax.annotate("", xy=(9.8, yd), xytext=(8.2, 4.2),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
        ax.text(9.9, yd, lbl, ha="left", va="center",
                fontsize=8.5, color=col, fontweight="bold", linespacing=1.4)
    row = df.iloc[0]
    ax.text(5.5, 0.9,
            f"Example (Exp 1  Test=Si)   "
            f"Acc={float(row['accuracy'])*100:.1f}%   AUC={float(row['roc_auc']):.2f}   Members=5",
            ha="center", va="center", fontsize=8.5, color="#aaaacc",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#1a1d27", edgecolor="#44444488"))
    ax.set_title("Deep Ensemble - 5 Independently Initialized Models (Seeds 42-46)\n"
                 "Model disagreement signals epistemic uncertainty",
                 fontsize=13, fontweight="bold", pad=8)
    _save(fig, OUT["unc"] / "deep_ensemble_prediction")

def fig_method_comparison():
    df = _load("ens_cmp")
    if df is None:
        _skip("deep_ensemble_comparison.png", "deep_ensemble_comparison.csv not found"); return
    method_order = ["Softmax", "Temperature Scaling", "EDL", "MC Dropout", "Deep Ensemble"]
    metrics = ["accuracy", "roc_auc", "nll", "brier", "ece"]
    labels  = ["Accuracy", "ROC-AUC", "NLL (lower=better)", "Brier", "ECE"]
    agg = df.groupby("method")[metrics].mean().reset_index()
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    for ax, metric, lbl in zip(axes, metrics, labels):
        vals, cols, mths = [], [], []
        for m in method_order:
            row = agg[agg["method"] == m]
            if row.empty or metric not in row.columns: continue
            vals.append(float(row[metric].iloc[0]))
            cols.append(S["method"].get(m, S["a1"])); mths.append(m)
        if not vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes); continue
        ax.bar(range(len(mths)), vals, color=cols, alpha=0.85,
               edgecolor="#ffffff22", width=0.55)
        ax.set_xticks(range(len(mths)))
        ax.set_xticklabels(mths, rotation=35, ha="right", fontsize=7.5)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        mx = max(vals)
        for j, val in enumerate(vals):
            ax.text(j, val + mx * 0.02, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7)
    fig.suptitle("Method Comparison - Macro-Average Across 3 LOCO Folds\n"
                 "Softmax / Temp Scaling / EDL / MC Dropout / Deep Ensemble",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, OUT["unc"] / "deep_ensemble_comparison")

# ===========================================================================
# 8. CALIBRATION
# ===========================================================================
def fig_calibration_metrics():
    df = _load("cal")
    if df is None:
        _skip("calibration_metrics.png", "calibration_results.csv not found"); return
    df = df[df["experiment"] != "MACRO_AVERAGE"]
    experiments = df["experiment"].unique()
    cohorts = []
    for e in experiments:
        r = df[df["experiment"] == e]
        cohorts.append(str(r["test_cohort"].iloc[0]) if not r.empty else str(e))
    metrics = ["nll", "brier", "ece"]
    labels  = ["NLL (lower=better)", "Brier (lower=better)", "ECE (lower=better)"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric, lbl in zip(axes, metrics, labels):
        if metric not in df.columns:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes); continue
        x = np.arange(len(experiments))
        for i, (method, col) in enumerate([
            ("softmax",             S["method"]["Softmax"]),
            ("temperature_scaling", S["method"]["Temperature Scaling"]),
        ]):
            vals = []
            for exp in experiments:
                row = df[(df["experiment"] == exp) & (df["method"] == method)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else 0)
            ax.bar(x + (i - 0.5) * 0.32, vals, 0.32, color=col, alpha=0.85,
                   label=method.replace("_", " ").title(), edgecolor="#ffffff22")
        ax.set_xticks(x); ax.set_xticklabels(cohorts, fontsize=9)
        ax.set_xlabel("Test Cohort", fontsize=9)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Calibration Metrics by LOCO Fold\nSoftmax vs Temperature Scaling",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, OUT["cal"] / "calibration_metrics")

def fig_reliability_diagram():
    df = _load("cal_subj")
    if df is None:
        _skip("reliability_diagram.png", "calibration_subject_predictions.csv not found"); return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, method in zip(axes, ["softmax", "temperature_scaling"]):
        sub = df[df["method"] == method] if "method" in df.columns else df
        if "P_PD" not in sub.columns or "true_label" not in sub.columns:
            ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray"); continue
        probs  = sub["P_PD"].values
        truths = sub["true_label"].values
        edges  = np.linspace(0, 1, 11)
        bconfs, baccs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                bconfs.append(probs[mask].mean())
                baccs.append(truths[mask].mean())
        ax.plot([0, 1], [0, 1], "w--", alpha=0.5, lw=1.5, label="Perfect calibration")
        col = S["method"].get("Softmax" if method == "softmax" else "Temperature Scaling", S["a1"])
        ax.bar(bconfs, baccs, width=0.08, alpha=0.6, color=col,
               edgecolor="#ffffff33", label=method.replace("_", " ").title())
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted P(PD)", fontsize=9)
        ax.set_ylabel("Observed Accuracy", fontsize=9)
        ax.set_title(method.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Reliability Diagram (Calibration Curve)\n"
                 "Predicted confidence vs observed accuracy - subject level",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, OUT["cal"] / "reliability_diagram")


# ===========================================================================
# 9. CORRUPTION / SHIFT
# ===========================================================================
def fig_corruption():
    df = _load("corr")
    if df is None:
        _skip("corruption_*.png", "corruption_shift_results.csv not found"); return
    metrics = [
        ("accuracy",         "Accuracy",           "corruption_accuracy",    True),
        ("nll",              "Neg Log-Likelihood", "corruption_nll",         False),
        ("ece",              "ECE",                "corruption_ece",         False),
        ("mean_uncertainty", "Mean Uncertainty",   "corruption_uncertainty", False),
    ]
    ctypes  = df["corruption_type"].unique() if "corruption_type" in df.columns else []
    clabels = {"gaussian_noise": "Gaussian Noise", "amplitude_scaling": "Amplitude Scaling",
               "temporal_masking": "Temporal Masking", "channel_dropout": "Channel Dropout"}
    methods = ["Softmax", "Temperature Scaling", "EDL", "MC Dropout"]
    for mc, ylabel, fname, pct in metrics:
        if mc not in df.columns:
            _skip(f"{fname}.png", f"Column {mc} missing"); continue
        ncols = max(1, len(ctypes))
        fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
        axlist = [axes] if ncols == 1 else list(axes)
        for ax, ctype in zip(axlist, ctypes):
            sub = df[(df["corruption_type"] == ctype) & (df["method"].isin(methods))]
            agg = sub.groupby(["severity", "method"])[mc].mean().reset_index()
            for method in methods:
                md = agg[agg["method"] == method].sort_values("severity")
                if md.empty: continue
                vals = md[mc].values * (100 if pct else 1)
                ax.plot(md["severity"].values, vals, "o-", lw=2, ms=5,
                        label=method, color=S["method"].get(method, S["a1"]))
            ax.set_title(clabels.get(ctype, ctype), fontsize=10, fontweight="bold")
            ax.set_xlabel("Corruption Severity (0=Clean -> 4=Severe)", fontsize=9)
            ax.set_ylabel(f"{ylabel} (%)" if pct else ylabel, fontsize=9)
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_xticklabels(["0\nClean", "1", "2", "3", "4\nSevere"])
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Corruption Stress Test - {ylabel}\nAveraged across LOCO folds",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        _save(fig, OUT["cohort"] / fname)


# ===========================================================================
# 10. SILENT CONFIDENCE
# ===========================================================================
def fig_silent_confidence():
    df = _load("corr")
    if df is None:
        _skip("silent_confidence.png", "corruption_shift_results.csv not found"); return
    sub = pd.DataFrame()
    if "corruption_type" in df.columns and "method" in df.columns:
        sub = df[(df["corruption_type"] == "gaussian_noise") &
                 (df["method"] == "Softmax")].groupby("severity")[
                     ["accuracy", "mean_uncertainty"]].mean().reset_index()
    if sub.empty or "accuracy" not in sub.columns:
        _skip("silent_confidence.png", "Cannot extract required rows"); return
    sevs = sub["severity"].values
    accs = sub["accuracy"].values * 100
    if "mean_uncertainty" not in sub.columns:
        _skip("silent_confidence.png", "mean_uncertainty column missing"); return
    uncs = sub["mean_uncertainty"].values
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    l1, = ax1.plot(sevs, accs,  "o-",  color=S["a1"], lw=2.5, ms=7, label="Accuracy (%)")
    l2, = ax2.plot(sevs, uncs,  "s--", color=S["a2"], lw=2.5, ms=7, label="Mean Uncertainty")
    ax1.set_xlabel("Corruption Severity (0=Clean -> 4=Severe)", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", fontsize=11, color=S["a1"])
    ax2.set_ylabel("Mean Uncertainty", fontsize=11, color=S["a2"])
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax1.set_xticklabels(["0\n(Clean)", "1", "2", "3", "4\n(Severe)"])
    if len(sevs) > 2:
        ax1.axvspan(sevs[1] - 0.3, sevs[-2] + 0.3, alpha=0.10, color=S["a4"])
    ax1.legend([l1, l2], [l.get_label() for l in [l1, l2]], fontsize=9, loc="lower left")
    ax1.set_title("Silent Confidence: Accuracy vs Uncertainty Under Corruption\n"
                  "Gaussian Noise  Softmax  - observed experimental behavior",
                  fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    _save(fig, OUT["unc"] / "silent_confidence")

# ===========================================================================
# 11. CONFORMAL PREDICTION
# ===========================================================================
def fig_conformal_coverage():
    df = _load("conf")
    if df is None:
        _skip("conformal_coverage.png / conformal_set_size.png",
              "conformal_results.csv not found"); return
    alphas  = sorted(df["alpha"].unique())
    exp_map = {1: "Si", 2: "Ju", 3: "Ga"}
    x = np.arange(len(alphas)); w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (exp, cohort) in enumerate(exp_map.items()):
        col  = S["cohort"].get(cohort, S["a1"])
        vals = []
        for alpha in alphas:
            row = df[(df["experiment"] == exp) & (np.isclose(df["alpha"], alpha))]
            vals.append(float(row["empirical_coverage"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (i - 1) * w, vals, w, color=col, alpha=0.85,
               label=f"Test={cohort}", edgecolor="#ffffff22")
    for alpha in alphas:
        ax.axhline(1 - alpha, color="white", ls=":", alpha=0.3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"a={a:.2f}\n(target={(1-a):.0%})" for a in alphas])
    ax.set_ylabel("Empirical Coverage", fontsize=11)
    ax.set_xlabel("Miscoverage Level (alpha)", fontsize=11)
    ax.set_title("Conformal Prediction - Empirical Coverage\nDotted lines = target coverage",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, OUT["conf"] / "conformal_coverage")

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for i, (exp, cohort) in enumerate(exp_map.items()):
        col  = S["cohort"].get(cohort, S["a1"])
        vals = []
        for alpha in alphas:
            row = df[(df["experiment"] == exp) & (np.isclose(df["alpha"], alpha))]
            vals.append(float(row["avg_set_size"].iloc[0]) if not row.empty else np.nan)
        ax2.bar(x + (i - 1) * w, vals, w, color=col, alpha=0.85,
                label=f"Test={cohort}", edgecolor="#ffffff22")
    ax2.set_xticks(x); ax2.set_xticklabels([f"a={a:.2f}" for a in alphas])
    ax2.set_ylabel("Average Set Size", fontsize=11)
    ax2.set_xlabel("Miscoverage Level (alpha)", fontsize=11)
    ax2.set_ylim(0, 2.2)
    ax2.axhline(1.0, color="white", ls=":", alpha=0.4, label="Singleton set")
    ax2.axhline(2.0, color=S["a2"], ls=":", alpha=0.4, label="Both classes")
    ax2.set_title("Conformal Prediction - Average Set Size\nLarger alpha -> more singletons",
                  fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, axis="y", alpha=0.3)
    _save(fig2, OUT["conf"] / "conformal_set_size")

def fig_conformal_corruption():
    df = _load("conf_corr")
    if df is None:
        _skip("conformal_corruption.png", "conformal_corruption_coverage.csv not found"); return
    ctypes = df["corruption_type"].unique() if "corruption_type" in df.columns else []
    cmap   = {1: "Si", 2: "Ju", 3: "Ga"}
    ncols  = max(1, len(ctypes))
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    axlist = [axes] if ncols == 1 else list(axes)
    for ax, ctype in zip(axlist, ctypes):
        for exp, cohort in cmap.items():
            col = S["cohort"].get(cohort, S["a1"])
            sub = df[(df["experiment"] == exp) & (df["corruption_type"] == ctype)
                     ].sort_values("severity")
            if sub.empty: continue
            ax.plot(sub["severity"].values, sub["empirical_coverage"].values,
                    "o-", color=col, lw=2, ms=6, label=f"Test={cohort}")
        ax.axhline(0.90, color="white", ls="--", alpha=0.4, label="Target 90%")
        ax.set_title(ctype.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("Corruption Severity", fontsize=9)
        ax.set_ylabel("Empirical Coverage", fontsize=9)
        ax.set_ylim(0, 1.15); ax.set_xticks([0, 1, 2, 3, 4])
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Conformal Coverage Under Input Corruption  (alpha=0.10, target=90%)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, OUT["conf"] / "conformal_corruption")

def fig_conformal_explained():
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    cases = [
        (1.7,  "{PD}",          S["a2"],   "High evidence for PD\nsingleton set",   "P_PD=0.94  P_HC=0.06"),
        (4.3,  "{Healthy, PD}", S["a4"],   "Ambiguous / uncertain\nboth included",  "P_PD=0.58  P_HC=0.42"),
        (6.9,  "{Healthy}",     S["a3"],   "High evidence for HC\nsingleton set",   "P_PD=0.03  P_HC=0.97"),
        (9.5,  "Empty set",     "#888899", "Very low scores\n(rare edge case)",     "P_PD=0.51  P_HC=0.49"),
    ]
    for cx, lbl, col, desc, prob in cases:
        ax.add_patch(FancyBboxPatch((cx - 1.25, 3.3), 2.5, 1.5,
            boxstyle="round,pad=0.12", linewidth=2,
            edgecolor=col, facecolor=col + "33"))
        ax.text(cx, 4.3, lbl, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col)
        ax.text(cx, 3.7, prob, ha="center", va="center",
                fontsize=7.5, color="#ccccdd", style="italic")
        ax.text(cx, 2.65, desc, ha="center", va="center",
                fontsize=8.5, color=S["text"], linespacing=1.4)
    ax.text(5.5, 1.5,
            "Conformal prediction guarantees >= 1-alpha coverage on the calibration distribution",
            ha="center", va="center", fontsize=9.5, color=S["a4"], fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#3a3010",
                      edgecolor=S["a4"] + "88", lw=1))
    ax.text(5.5, 0.75,
            "The prediction set contains all plausible classes for the given confidence level",
            ha="center", va="center", fontsize=8.5, color="#aaaacc")
    ax.set_title("Conformal Prediction - Possible Output Sets\nalpha=0.10  ->  >=90% coverage guarantee",
                 fontsize=13, fontweight="bold", pad=8)
    _save(fig, OUT["conf"] / "conformal_prediction_explained")

# ===========================================================================
# 12. RELIABILITY
# ===========================================================================
def fig_reliability_results():
    df = _load("rel")
    if df is None:
        _skip("reliability/*.png", "reliability_results.csv not found"); return
    c_order = ["Si", "Ju", "Ga"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(c_order)); w = 0.32
    for i, (col_n, lbl, col) in enumerate([
        ("raw_accuracy",      "Raw Accuracy",      S["a1"]),
        ("accepted_accuracy", "Accepted Accuracy", S["a3"]),
    ]):
        vals = []
        for cohort in c_order:
            row = df[df["test_cohort"] == cohort]
            vals.append(float(row[col_n].iloc[0]) * 100
                        if (not row.empty and col_n in row.columns) else 0)
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=col, alpha=0.85,
                      label=lbl, edgecolor="#ffffff22")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Test={c}" for c in c_order], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Raw Accuracy vs Accepted-Only Accuracy\n"
                 "Reliability layer filters uncertain predictions",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, OUT["rel"] / "raw_vs_accepted_accuracy")

    for col_n, ylabel, fname in [
        ("coverage",           "Coverage (% Accepted)",  "coverage_by_cohort"),
        ("error_capture_rate", "Error Capture Rate (%)", "error_capture_by_cohort"),
    ]:
        if col_n not in df.columns:
            _skip(f"{fname}.png", f"Column {col_n} missing"); continue
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        vals, colors = [], []
        for cohort in c_order:
            row = df[df["test_cohort"] == cohort]
            vals.append(float(row[col_n].iloc[0]) * 100 if not row.empty else 0)
            colors.append(S["cohort"].get(cohort, S["a1"]))
        bars = ax2.bar(c_order, vals, color=colors, alpha=0.85,
                       width=0.5, edgecolor="#ffffff22")
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom",
                     fontsize=11, fontweight="bold", color="white")
        ax2.set_ylabel(ylabel, fontsize=11)
        ax2.set_xlabel("Test Cohort", fontsize=11)
        ax2.set_title(f"{ylabel} by Cohort", fontsize=13, fontweight="bold")
        ax2.set_ylim(0, 105); ax2.grid(True, axis="y", alpha=0.3)
        _save(fig2, OUT["rel"] / fname)

def fig_flag_reasons():
    df = _load("rel_subj")
    if df is None:
        _skip("flag_reason_breakdown.png",
              "reliability_subject_decisions.csv not found"); return
    flagged = df[df["reliability_status"] == "FLAG"]
    if flagged.empty:
        _skip("flag_reason_breakdown.png", "No flagged subjects"); return
    ALL_R = ["HIGH_UNCERTAINTY", "LOW_CONFIDENCE", "AMBIGUOUS_CONFORMAL_SET",
             "EMPTY_CONFORMAL_SET", "CONFORMAL_POINT_CONTRADICTION"]
    SHORT = {
        "HIGH_UNCERTAINTY":            "High\nUncertainty",
        "LOW_CONFIDENCE":              "Low\nConfidence",
        "AMBIGUOUS_CONFORMAL_SET":     "Ambiguous\nSet",
        "EMPTY_CONFORMAL_SET":         "Empty\nSet",
        "CONFORMAL_POINT_CONTRADICTION": "Contradiction",
    }
    RCOLS = dict(zip(ALL_R, [S["a2"], S["a1"], S["a4"], S["a5"], S["a3"]]))
    cohorts = sorted(flagged["test_cohort"].unique())
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 6))
    axlist = [axes] if len(cohorts) == 1 else list(axes)
    for ax, cohort in zip(axlist, cohorts):
        sub = flagged[flagged["test_cohort"] == cohort]
        counts = {r: 0 for r in ALL_R}
        for rs in sub["flag_reasons"].dropna():
            for r in str(rs).split(";"):
                r = r.strip()
                if r in counts:
                    counts[r] += 1
        nz = [(r, v) for r, v in counts.items() if v > 0]
        if not nz:
            ax.text(0.5, 0.5, "No flags", ha="center", va="center",
                    transform=ax.transAxes); continue
        reasons, vals = zip(*nz)
        ax.bar(range(len(vals)), vals,
               color=[RCOLS.get(r, S["a1"]) for r in reasons],
               alpha=0.85, edgecolor="#ffffff22", width=0.55)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([SHORT.get(r, r) for r in reasons],
                           fontsize=8, rotation=20, ha="right")
        ax.set_title(f"Test Cohort: {cohort}", fontsize=10, fontweight="bold",
                     color=S["cohort"].get(cohort, "white"))
        ax.set_ylabel("Flagged Subjects (count)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        for j, val in enumerate(vals):
            ax.text(j, val + 0.1, str(val), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    fig.suptitle("FLAG Reason Breakdown by Cohort\n"
                 "Subjects may have multiple flag reasons",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, OUT["rel"] / "flag_reason_breakdown")

def fig_subject_distributions():
    df = _load("rel_subj")
    if df is None:
        _skip("subject_*.png", "reliability_subject_decisions.csv not found"); return
    for metric, ylabel, fname in [
        ("confidence",  "Prediction Confidence (max probability)",  "subject_confidence_distribution"),
        ("uncertainty", "Prediction Uncertainty (1-Confidence)",    "subject_uncertainty_distribution"),
    ]:
        if metric not in df.columns:
            _skip(f"{fname}.png", f"Column {metric} missing"); continue
        cohorts = sorted(df["test_cohort"].unique())
        fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 5))
        axlist = [axes] if len(cohorts) == 1 else list(axes)
        for ax, cohort in zip(axlist, cohorts):
            sub  = df[df["test_cohort"] == cohort]
            acc  = sub[sub["reliability_status"] == "ACCEPT"][metric].values
            flag = sub[sub["reliability_status"] == "FLAG"][metric].values
            ax.hist(acc,  bins=15, alpha=0.7, color=S["a3"],
                    label=f"ACCEPT (n={len(acc)})",  edgecolor="#ffffff22")
            ax.hist(flag, bins=15, alpha=0.7, color=S["a2"],
                    label=f"FLAG (n={len(flag)})", edgecolor="#ffffff22")
            ax.set_title(f"Test Cohort: {cohort}", fontsize=10,
                         fontweight="bold", color=S["cohort"].get(cohort, "white"))
            ax.set_xlabel(ylabel, fontsize=9)
            ax.set_ylabel("Number of Subjects", fontsize=9)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.suptitle(f"{ylabel}\nACCEPT vs FLAG", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        _save(fig, OUT["rel"] / fname)

# ===========================================================================
# 13. DECISION FLOW
# ===========================================================================
def fig_decision_flow():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")
    steps = [
        ("Model Prediction",   11.0, "#2d6a9f",
         "P(PD), P(HC) via temperature-scaled evidential model"),
        ("Confidence Check",    9.4, "#1a5740",
         "Confidence = max(P_PD, P_HC)   Flag if Confidence < tau_conf"),
        ("Uncertainty Check",   7.8, "#5a1a72",
         "Uncertainty = 1 - Confidence   Flag if Uncertainty > tau_unc"),
        ("Conformal Set Check", 6.2, "#725a1a",
         "Build prediction set at alpha=0.10   Flag if |set| > 1 or |set| = 0"),
    ]
    bx, bw, bh = 2.0, 6.0, 0.80
    for i, (lbl, yc, col, sub) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((bx, yc - bh/2), bw, bh,
            boxstyle="round,pad=0.09", linewidth=1.5,
            edgecolor=col + "cc", facecolor=col + "55"))
        ax.text(bx + bw/2, yc + 0.12, lbl, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(bx + bw/2, yc - 0.24, sub, ha="center", va="center",
                fontsize=7.5, color="#ccccdd", style="italic")
        if i < len(steps) - 1:
            nyc = steps[i + 1][1]
            ax.annotate("", xy=(5.0, nyc + bh/2 + 0.04),
                        xytext=(5.0, yc - bh/2 - 0.04),
                        arrowprops=dict(arrowstyle="->", color="#aaaacc", lw=1.8))
    ax.annotate("", xy=(5.0, 5.15), xytext=(5.0, 5.82),
                arrowprops=dict(arrowstyle="->", color="#aaaacc", lw=1.8))
    diamond = plt.Polygon([(5.0, 4.6), (6.2, 5.15), (5.0, 5.7), (3.8, 5.15)],
                          closed=True, facecolor="#2a2d3a",
                          edgecolor="#aaaacc", lw=1.5)
    ax.add_patch(diamond)
    ax.text(5.0, 5.15, "Any flag\ntriggered?", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    for xp, lbl, col in [(2.5, "ACCEPT\nReliable", "#1a7a3a"),
                          (7.5, "FLAG\nAbstain",    "#7a2020")]:
        ax.add_patch(FancyBboxPatch((xp - 1.5, 2.2), 3.0, 1.2,
            boxstyle="round,pad=0.1", linewidth=2,
            edgecolor=col, facecolor=col + "44"))
        ax.text(xp, 2.82, lbl, ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=S["a3"] if "ACCEPT" in lbl else S["a2"], linespacing=1.4)
    ax.annotate("NO",  xy=(2.5, 3.4), xytext=(3.8, 4.9),
                arrowprops=dict(arrowstyle="->", color=S["a3"], lw=2.0),
                fontsize=9, color=S["a3"], fontweight="bold", ha="right", va="center")
    ax.annotate("YES", xy=(7.5, 3.4), xytext=(6.2, 4.9),
                arrowprops=dict(arrowstyle="->", color=S["a2"], lw=2.0),
                fontsize=9, color=S["a2"], fontweight="bold", ha="left", va="center")
    ax.text(5.0, 1.15,
            "MODEL RELIABILITY / ABSTENTION INDICATOR\n"
            "NOT a clinical diagnosis or medical decision system.",
            ha="center", va="center", fontsize=9, color=S["a4"],
            fontweight="bold", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#3a3010",
                      edgecolor=S["a4"] + "88", lw=1))
    ax.set_title("Reliability / Abstention Decision Flow\n"
                 "Three-layer check: Confidence -> Uncertainty -> Conformal Set",
                 fontsize=13, fontweight="bold", pad=8)
    _save(fig, OUT["rel"] / "reliability_decision_flow")


# ===========================================================================
# 14. SYSTEM OVERVIEW
# ===========================================================================
def fig_system_overview():
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 14); ax.set_ylim(0, 12); ax.axis("off")
    pipeline = [
        (7.0, 11.2, "GAIT SIGNAL\n16 VGRF channels  100 Hz",        "#1e3a5f", 2.8, 0.80),
        (7.0,  9.8, "PREPROCESSING\nNormalise  Window  Overlap",     "#1a5740", 2.8, 0.80),
        (7.0,  8.2, "CNN + BiLSTM MODEL\n128-dim Gait Embedding",    "#3d1f72", 2.8, 0.80),
        (7.0,  6.6, "PD PROBABILITY\nEvidential Head",               "#3d1f72", 2.8, 0.80),
        (4.5,  5.1, "CALIBRATION\nTemperature Scaling",              "#725a1a", 2.4, 0.75),
        (9.5,  5.1, "UNCERTAINTY\n1 - Confidence",                   "#5a1a72", 2.4, 0.75),
        (7.0,  3.6, "CONFORMAL SET\nalpha=0.10",                     "#5a3a00", 2.8, 0.80),
        (7.0,  2.1, "RELIABILITY LAYER\ntau_conf  tau_unc  q_hat",   "#2a5a1a", 2.8, 0.80),
    ]
    for cx, cy, lbl, col, bw, bh in pipeline:
        ax.add_patch(FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
            boxstyle="round,pad=0.09", linewidth=1.5,
            edgecolor=col + "cc", facecolor=col + "88"))
        ax.text(cx, cy, lbl, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", linespacing=1.35)
    for x1, y1, x2, y2 in [
        (7, 10.8, 7, 10.2), (7, 9.4, 7, 8.62), (7, 7.78, 7, 7.0),
        (7, 6.2, 5.7, 5.5), (7, 6.2, 8.3, 5.5),
        (5.7, 4.73, 7, 4.0), (8.3, 4.73, 7, 4.0), (7, 3.2, 7, 2.5),
    ]:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#aaaacc", lw=1.5))
    for xp, lbl, col in [(4.5, "ACCEPT", "#1a7a3a"), (9.5, "FLAG", "#7a2020")]:
        ax.add_patch(FancyBboxPatch((xp - 1.1, 0.5), 2.2, 0.9,
            boxstyle="round,pad=0.1", linewidth=2,
            edgecolor=col, facecolor=col + "44"))
        ax.text(xp, 0.95, lbl, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col)
        ax.annotate("", xy=(xp, 1.4), xytext=(7.0, 1.7),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.0))
    for cx, cy, lbl, col in [
        (1.0, 8.8, "LOCO\nEvaluation\n3 folds",  S["a1"]),
        (1.0, 7.0, "Multi-Seed\n42-46",           S["a3"]),
        (1.0, 5.2, "Deep\nEnsemble\n5 members",   S["a2"]),
        (13,  8.8, "MC\nDropout",                 S["a5"]),
        (13,  7.0, "Corruption\nTesting",          S["a4"]),
        (13,  5.2, "Conformal\nPrediction",        S["a1"]),
    ]:
        ax.text(cx, cy, lbl, ha="center", va="center", fontsize=8,
                fontweight="bold", color=col, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor=col + "22", edgecolor=col + "66", lw=1))
    ax.set_title("Complete Gait Parkinson's Disease Detection System\n"
                 "From raw VGRF signal to reliability-aware prediction",
                 fontsize=14, fontweight="bold", pad=10)
    _save(fig, OUT["arch"] / "final_system_overview")


# ===========================================================================
# 15. PROJECT STORY
# ===========================================================================
def fig_project_story():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")
    stages = [
        ("1. DETECT",     "Gait signal\n-> PD prediction",             "#2d6a9f"),
        ("2. GENERALIZE", "LOCO evaluation\n-> unseen cohort",         "#1a5740"),
        ("3. CALIBRATE",  "Raw probability\n-> calibrated probability", "#725a1a"),
        ("4. UNCERTAINTY","MC Dropout / Ensemble\n-> uncertainty",      "#5a1a72"),
        ("5. ABSTAIN",    "ACCEPT / FLAG\n-> reliable subset",          "#1a5a2d"),
    ]
    cxs = np.linspace(1.4, 12.6, len(stages))
    for i, (cx, (title, body, col)) in enumerate(zip(cxs, stages)):
        ax.add_patch(FancyBboxPatch((cx - 1.1, 0.8), 2.2, 3.0,
            boxstyle="round,pad=0.12", linewidth=2,
            edgecolor=col, facecolor=col + "44"))
        ax.text(cx, 3.5, title.split(".")[0] + ".", ha="center", va="center",
                fontsize=20, fontweight="bold", color=col + "cc")
        ax.text(cx, 2.9, title.split(". ", 1)[1], ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(cx, 2.1, body, ha="center", va="center",
                fontsize=8.5, color="#ccccdd", linespacing=1.5)
        if i < len(stages) - 1:
            ncx = cxs[i + 1]
            ax.annotate("", xy=(ncx - 1.1, 2.3), xytext=(cx + 1.1, 2.3),
                        arrowprops=dict(arrowstyle="->", color="#aaaacc", lw=2.0))
    ax.set_title("Project Story: Can Uncertainty Warn Us?\n"
                 "Cohort-Shift Reliability of Parkinson's Disease Classifiers",
                 fontsize=13, fontweight="bold", pad=6)
    _save(fig, OUT["arch"] / "project_story")


# ===========================================================================
# 16. INFERENCE FLOW
# ===========================================================================
def fig_inference_flow():
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(0, 11); ax.set_ylim(0, 14); ax.axis("off")
    steps = [
        (5.5, 13.0, "Upload gait .txt recording",          "#1e3a5f",
         "raw 16-column VGRF file from PhysioNet format"),
        (5.5, 11.6, "16-channel VGRF signal",               "#1e3a5f",
         "L1-L8   R1-R8   100 Hz"),
        (5.5, 10.2, "Windowing",                            "#1a5740",
         "5-sec windows   50% overlap   N windows total"),
        (5.5,  8.8, "Frozen pretrained model",              "#3d1f72",
         "GaitCNNBiLSTM   seed=42   LOCO fold checkpoint"),
        (5.5,  7.4, "P(PD) per window -> Recording mean",   "#3d1f72",
         "Window-level forward pass   mean aggregation"),
        (5.5,  6.0, "Temperature scaling (calibration)",    "#725a1a",
         "Learned T per fold   improves probability reliability"),
        (5.5,  4.6, "Uncertainty = 1 - Confidence",         "#5a1a72",
         "Confidence = max(P_PD, P_HC)"),
        (5.5,  3.2, "Conformal prediction set",             "#5a3a00",
         "alpha=0.10   {PD}, {Healthy}, or {Healthy, PD}"),
        (5.5,  1.8, "Reliability decision",                 "#1a5a2d",
         "tau_conf  tau_unc  q_hat  ->  ACCEPT / FLAG"),
    ]
    bx, bw, bh = 2.0, 7.0, 0.72
    for i, (cx, cy, lbl, col, sub) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((bx, cy - bh/2), bw, bh,
            boxstyle="round,pad=0.08", linewidth=1,
            edgecolor=col + "cc", facecolor=col + "88"))
        ax.text(cx, cy + 0.10, lbl, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white")
        ax.text(cx, cy - 0.25, sub, ha="center", va="center",
                fontsize=7.5, color="#ccccdd", style="italic")
        if i < len(steps) - 1:
            ncy = steps[i + 1][1]
            ax.annotate("", xy=(cx, ncy + bh/2 + 0.04),
                        xytext=(cx, cy - bh/2 - 0.04),
                        arrowprops=dict(arrowstyle="->", color="#9999bb", lw=1.5))
    ax.add_patch(FancyBboxPatch((0.3, 0.08), 10.4, 0.95,
        boxstyle="round,pad=0.1", linewidth=1.5,
        edgecolor="#555566", facecolor="#1a1d27"))
    ax.text(5.5, 0.60,
            "Example Output | Prediction: [PD or Healthy]  "
            "P(PD): XX%  Uncertainty: XX%  Conformal: {PD}  Reliability: ACCEPT",
            ha="center", va="center", fontsize=7.5, color="#aaaacc", style="italic")
    ax.text(5.5, 0.22,
            "XX = placeholders  Actual values depend on the input recording",
            ha="center", va="center", fontsize=7.0, color="#666688", style="italic")
    ax.set_title("Individual Gait Inference - Zero-Retraining Pipeline\n"
                 "Frozen checkpoints and precomputed calibration thresholds",
                 fontsize=13, fontweight="bold", pad=8)
    _save(fig, OUT["inf"] / "live_inference_flow")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("\n" + "=" * 64)
    print("  Gait Parkinson's Disease - Visualization Generator")
    print("=" * 64)

    print("\n--- Architecture ---")
    fig_model_architecture()
    fig_system_overview()
    fig_project_story()

    print("\n--- Data Pipeline ---")
    fig_pipeline()
    fig_gait_signal()
    fig_windowing()

    print("\n--- Cohort Shift ---")
    fig_loco_protocol()
    fig_corruption()

    print("\n--- Performance ---")
    fig_loco_performance()
    fig_multiseed_performance()
    fig_multiseed_uncertainty()

    print("\n--- Calibration ---")
    fig_calibration_metrics()
    fig_reliability_diagram()

    print("\n--- Uncertainty ---")
    fig_ensemble_schematic()
    fig_method_comparison()
    fig_silent_confidence()

    print("\n--- Conformal Prediction ---")
    fig_conformal_coverage()
    fig_conformal_corruption()
    fig_conformal_explained()

    print("\n--- Reliability / Abstention ---")
    fig_reliability_results()
    fig_flag_reasons()
    fig_decision_flow()
    fig_subject_distributions()

    print("\n--- Inference ---")
    fig_inference_flow()

    print("\n" + "=" * 64)
    print(f"  Generated : {len(GEN)} files")
    print(f"  Skipped   : {len(SKIP)} figures")
    print("=" * 64)
    if GEN:
        print("\nGenerated:")
        for p in GEN:
            try:
                print(f"  {Path(p).relative_to(_ROOT_DIR)}")
            except Exception:
                print(f"  {p}")
    if SKIP:
        print("\nSkipped (missing source data):")
        for lbl, reason in SKIP:
            print(f"  -- {lbl}: {reason}")
    print()


if __name__ == "__main__":
    main()
