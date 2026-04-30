"""
ui_frontend.py
-------------
Gradio frontend for the Cybersecurity Intrusion Detection ML pipeline.

Tabs:
  1. Live Prediction      - Score a single network session from a form.
  2. Batch Prediction     - Upload a CSV of sessions, get predictions in bulk.
  3. Model Comparison     - Side-by-side metrics for every trained model
                            (pulled from MLflow tracking DB).
  4. EDA & Explainability - The plots produced during analysis (class
                            balance, correlations, ROC, confusion matrix,
                            SHAP, etc.).
  5. About                - Project description + dataset cheatsheet.

Run:
    python src/ui_frontend.py
The app starts at http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Compatibility shim for gradio 5.6.0 / gradio_client 1.5.x
# ---------------------------------------------------------------------------
# Three functions in `gradio_client.utils` walk JSON Schemas and assume every
# subschema is a dict. JSON Schema 2020-12 actually allows bool subschemas
# (e.g. `additionalProperties: false`), which Pydantic v2 — used internally
# by Gradio — happily emits. The unpatched code paths fail with either:
#     TypeError: argument of type 'bool' is not iterable                 (get_type)
#     APIInfoParseError: Cannot parse schema False                       (_json_schema_to_python_type)
#
# These are fixed in newer gradio_client builds, but pinning a newer
# gradio_client transitively pulls a huggingface_hub that removed HfFolder,
# breaking gradio.oauth instead. Patching here keeps every version working.
#
# Must be applied BEFORE `import gradio` so the wrapped functions are used
# when gradio enumerates API schemas at launch time.
def _patch_gradio_client_for_bool_schemas() -> None:
    try:
        from gradio_client import utils as _gcu
    except Exception:
        return  # gradio_client not installed yet — patch becomes a no-op

    # 1. `get_type(schema)` — return the JSON-Schema sentinel for non-dict.
    if hasattr(_gcu, "get_type"):
        _orig_get_type = _gcu.get_type

        def _safe_get_type(schema):
            if not isinstance(schema, dict):
                return "any"
            return _orig_get_type(schema)

        _gcu.get_type = _safe_get_type

    # 2. `_json_schema_to_python_type(schema, defs)` — return "Any" for non-dict.
    if hasattr(_gcu, "_json_schema_to_python_type"):
        _orig_jspt = _gcu._json_schema_to_python_type

        def _safe_jspt(schema, *args, **kwargs):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_jspt(schema, *args, **kwargs)

        _gcu._json_schema_to_python_type = _safe_jspt

    # 3. The public alias used by the API docs builder.
    if hasattr(_gcu, "json_schema_to_python_type"):
        _orig_pub = _gcu.json_schema_to_python_type

        def _safe_pub(schema, *args, **kwargs):
            if not isinstance(schema, dict):
                return "Any"
            return _orig_pub(schema, *args, **kwargs)

        _gcu.json_schema_to_python_type = _safe_pub


_patch_gradio_client_for_bool_schemas()

import gradio as gr

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend (works inside containers / servers)
import matplotlib.pyplot as plt

# --- Paths -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # repo root
MODEL_PATH   = ROOT / "models" / "model.pkl"
SCALER_PATH  = ROOT / "models" / "scaler.pkl"
OUTPUTS_DIR  = ROOT / "outputs"
MLFLOW_DBS   = [ROOT / "mlflow.db", ROOT / "src" / "mlflow.db",
                ROOT / "mlflow_tracking.db", ROOT / "src" / "mlflow_tracking.db"]

# --- Constants matching preprocessing.py ------------------------------------

NUM_COLS = [
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
]

# Order of columns expected by the saved model
# (matches preprocessing.run_pipeline -> add_features -> encode -> split)
EXPECTED_COLS = [
    "network_packet_size",
    "login_attempts",
    "session_duration",
    "ip_reputation_score",
    "failed_logins",
    "unusual_time_access",
    "risk_index",
    "login_speed",
    "protocol_type_ICMP",
    "protocol_type_TCP",
    "protocol_type_UDP",
    "encryption_used_AES",
    "encryption_used_DES",
    "encryption_used_None",
    "browser_type_Chrome",
    "browser_type_Edge",
    "browser_type_Firefox",
    "browser_type_Safari",
    "browser_type_Unknown",
]

# The threshold chosen during training (Voting dt+knn+lr).
DEFAULT_THRESHOLD = 0.40

# --- Load the ML artefacts once at startup ----------------------------------

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


MODEL = _load_pickle(MODEL_PATH)
SCALER = _load_pickle(SCALER_PATH)
MODEL_NAME = type(MODEL).__name__
if hasattr(MODEL, "named_estimators_"):
    parts = "+".join(MODEL.named_estimators_.keys())
    MODEL_LABEL = f"{MODEL_NAME} ({parts})"
else:
    MODEL_LABEL = MODEL_NAME


# --- Feature engineering helpers (mirror preprocessing.py) ------------------

def _build_feature_row(
    network_packet_size: int,
    protocol_type: str,
    login_attempts: int,
    session_duration: float,
    encryption_used: str,
    ip_reputation_score: float,
    failed_logins: int,
    browser_type: str,
    unusual_time_access: int,
) -> pd.DataFrame:
    """Build a single-row DataFrame with the exact columns the model expects."""
    risk_index  = (1 - ip_reputation_score) * failed_logins
    login_speed = login_attempts / (session_duration + 0.1)

    raw = {
        "network_packet_size": network_packet_size,
        "login_attempts":      login_attempts,
        "session_duration":    session_duration,
        "ip_reputation_score": ip_reputation_score,
        "failed_logins":       failed_logins,
        "unusual_time_access": int(unusual_time_access),
        "risk_index":          risk_index,
        "login_speed":         login_speed,
        "protocol_type_ICMP":  0,
        "protocol_type_TCP":   0,
        "protocol_type_UDP":   0,
        "encryption_used_AES": 0,
        "encryption_used_DES": 0,
        "encryption_used_None": 0,
        "browser_type_Chrome": 0,
        "browser_type_Edge":   0,
        "browser_type_Firefox":0,
        "browser_type_Safari": 0,
        "browser_type_Unknown":0,
    }
    raw[f"protocol_type_{protocol_type}"] = 1
    raw[f"encryption_used_{encryption_used}"] = 1
    raw[f"browser_type_{browser_type}"] = 1

    df = pd.DataFrame([raw])[EXPECTED_COLS]
    df[NUM_COLS] = SCALER.transform(df[NUM_COLS])
    return df


# --- Tab 1: Live prediction --------------------------------------------------

def predict_session(
    network_packet_size, protocol_type, login_attempts, session_duration,
    encryption_used, ip_reputation_score, failed_logins, browser_type,
    unusual_time_access, threshold,
):
    try:
        x = _build_feature_row(
            network_packet_size, protocol_type, login_attempts, session_duration,
            encryption_used, ip_reputation_score, failed_logins, browser_type,
            unusual_time_access,
        )
        proba_attack = float(MODEL.predict_proba(x)[0, 1])
        is_attack = proba_attack >= threshold

        verdict_md = (
            f"## ATTACK DETECTED\n"
            f"**Probability of intrusion: `{proba_attack:.2%}`**  \n"
            f"Threshold used: `{threshold:.2f}`"
            if is_attack else
            f"## Normal session\n"
            f"**Probability of intrusion: `{proba_attack:.2%}`**  \n"
            f"Threshold used: `{threshold:.2f}`"
        )

        label_data = {
            "Attack":  proba_attack,
            "Normal":  1.0 - proba_attack,
        }
        return verdict_md, label_data
    except Exception as e:
        return f"### Error\n```\n{e}\n```", {}


def reset_form():
    return 500, "TCP", 3, 120.0, "AES", 0.50, 2, "Chrome", 0, DEFAULT_THRESHOLD


LIVE_EXAMPLES = [
    [1500, "TCP", 9, 12.0,  "None", 0.05, 8, "Unknown", 1, DEFAULT_THRESHOLD],
    [400,  "TCP", 1, 220.0, "AES",  0.92, 0, "Chrome",  0, DEFAULT_THRESHOLD],
    [800,  "UDP", 4, 35.0,  "DES",  0.30, 5, "Firefox", 1, DEFAULT_THRESHOLD],
]


# --- Tab 2: Batch prediction -------------------------------------------------

def predict_batch(file, threshold):
    if file is None:
        return None, "Upload a CSV first."
    try:
        df_raw = pd.read_csv(file.name)
    except Exception as e:
        return None, f"Could not parse CSV: {e}"

    expected_inputs = {
        "network_packet_size", "protocol_type", "login_attempts",
        "session_duration", "encryption_used", "ip_reputation_score",
        "failed_logins", "browser_type", "unusual_time_access",
    }
    missing = expected_inputs - set(df_raw.columns)
    if missing:
        return None, (
            "CSV is missing required columns: "
            f"`{', '.join(sorted(missing))}`.\n\n"
            "Required columns: " + ", ".join(sorted(expected_inputs))
        )

    rows = []
    for _, row in df_raw.iterrows():
        rows.append(_build_feature_row(
            int(row["network_packet_size"]),
            str(row["protocol_type"]),
            int(row["login_attempts"]),
            float(row["session_duration"]),
            str(row["encryption_used"]) if pd.notna(row["encryption_used"]) else "None",
            float(row["ip_reputation_score"]),
            int(row["failed_logins"]),
            str(row["browser_type"]),
            int(row["unusual_time_access"]),
        ))
    X = pd.concat(rows, ignore_index=True)
    proba = MODEL.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    out = df_raw.copy()
    out["attack_probability"] = np.round(proba, 4)
    out["prediction"] = pred
    out["verdict"] = np.where(pred == 1, "ATTACK", "Normal")

    n = len(out)
    n_attack = int(pred.sum())
    summary = (
        f"**{n}** sessions scored.  \n"
        f"**{n_attack}** flagged as attacks ({n_attack / n:.1%}).  \n"
        f"Average attack probability: **{proba.mean():.2%}**.  \n"
        f"Threshold used: **{threshold:.2f}**."
    )
    return out, summary


# --- Tab 3: Model comparison -------------------------------------------------

FALLBACK_RUNS = [
    ("Voting Ensemble dt+knn+lr", 0.8134, 0.8007, 0.7860, 0.7933, 0.8731, 0.4029),
    ("Voting Ensemble dt+knn",    0.7966, 0.8042, 0.7563, 0.7795, 0.8761, 0.2964),
    ("Voting Ensemble rf+dt",     0.8145, 0.8019, 0.7871, 0.7944, 0.8785, 0.2650),
    ("Random Forest (Balanced)",  0.8045, 0.8077, 0.7673, 0.7870, 0.8786, 0.2400),
    ("Decision Tree (Balanced)",  0.7411, 0.8312, 0.6695, 0.7416, 0.8738, 0.2986),
]


def _read_runs_from_mlflow():
    for path in MLFLOW_DBS:
        if not path.exists():
            continue
        try:
            con = sqlite3.connect(str(path))
            cur = con.cursor()
            cur.execute(
                """
                SELECT r.run_uuid, r.name, r.start_time
                FROM runs r
                JOIN experiments e ON r.experiment_id = e.experiment_id
                WHERE e.name = 'cybersecurity_intrusion_detection'
                  AND r.status = 'FINISHED'
                ORDER BY r.start_time DESC
                """
            )
            rows = cur.fetchall()
            seen, ordered_runs = set(), []
            for run_id, name, _ in rows:
                if name in seen or name is None:
                    continue
                seen.add(name)
                ordered_runs.append((run_id, name))

            results = []
            for run_id, name in ordered_runs:
                cur.execute(
                    "SELECT key, value FROM latest_metrics WHERE run_uuid = ?",
                    (run_id,),
                )
                metrics = dict(cur.fetchall())
                if not any(k in metrics for k in ("test_accuracy", "test_recall", "test_auc")):
                    continue
                results.append((
                    name,
                    metrics.get("test_accuracy"),
                    metrics.get("test_recall"),
                    metrics.get("test_precision"),
                    metrics.get("test_f1_score"),
                    metrics.get("test_auc"),
                    metrics.get("optimal_threshold"),
                ))
            con.close()
            if results:
                return results
        except Exception:
            continue
    return []


def get_comparison_data():
    runs = _read_runs_from_mlflow()
    if runs:
        df = pd.DataFrame(runs, columns=[
            "Model", "Accuracy", "Recall", "Precision", "F1-score", "AUC", "Threshold",
        ])
    else:
        df = pd.DataFrame(
            FALLBACK_RUNS,
            columns=["Model", "Accuracy", "Recall", "Precision", "F1-score", "AUC", "Threshold"],
        )

    best_name = "Voting Ensemble dt+knn+lr"
    df = df.copy()
    df["Best?"] = np.where(df["Model"] == best_name, "* deployed", "")
    df = df.sort_values(by="AUC", ascending=False).reset_index(drop=True)
    df_pretty = df.round(4)

    chart_df = df_pretty.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Recall", "Precision", "F1-score", "AUC"],
        var_name="Metric",
        value_name="Score",
    )
    return df_pretty, chart_df, best_name


COMPARISON_CHART_PATH = OUTPUTS_DIR / "comparison_chart.png"


def build_comparison_chart(df_pretty: pd.DataFrame, save_path: Path = COMPARISON_CHART_PATH) -> Path:
    """Render the grouped bar chart to a PNG and return its path.

    Why a pre-rendered PNG instead of `gr.Plot(value=Figure)`?

    Gradio 5 serialises live matplotlib Figures more aggressively than v4 did
    and can trigger a frontend render loop when something in the figure state
    fails to round-trip (this surfaced specifically inside the Docker image).
    A static PNG goes through the exact same file-serving path as the EDA
    gallery, which we already know works.
    """
    metrics = ["Accuracy", "Recall", "Precision", "F1-score", "AUC"]
    models = list(df_pretty["Model"])
    n_models, n_metrics = len(models), len(metrics)

    x = np.arange(n_models)
    width = 0.8 / n_metrics
    palette = ["#2563eb", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444"]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, m in enumerate(metrics):
        offset = (i - n_metrics / 2) * width + width / 2
        bars = ax.bar(x + offset, df_pretty[m], width, label=m,
                      color=palette[i % len(palette)])
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color="#334155")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test-set metrics by model", fontsize=12, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", ncol=5, fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --- Tab 4: EDA & explainability --------------------------------------------

EDA_PLOTS = [
    ("eda_classes.png",      "Class balance: normal vs. attack"),
    ("eda_heatmap.png",      "Feature correlation heatmap"),
    ("eda_boxplots.png",     "Numeric features by class"),
    ("eda_categorical.png",  "Categorical feature distributions"),
]
EVAL_PLOTS = [
    ("roc_curve.png",        "ROC curve of the deployed model"),
    ("confusion_matrix.png", "Confusion matrix"),
    ("shap_summary.png",     "SHAP - global feature importance"),
    ("shap_waterfall.png",   "SHAP - per-prediction explanation"),
]


def _gallery_items(plots):
    return [
        (str(OUTPUTS_DIR / fn), title)
        for fn, title in plots
        if (OUTPUTS_DIR / fn).exists()
    ]


# --- UI ---------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {max-width: 1200px !important;}
#hero {
    background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    padding: 24px 28px;
    border-radius: 12px;
    margin-bottom: 12px;
}
#hero h1 {color: white; margin: 0;}
#hero p {color: #cbd5e1; margin: 6px 0 0 0;}
.metric-pill {
    background: #1e293b;
    color: #38bdf8;
    padding: 4px 10px;
    border-radius: 999px;
    margin-right: 6px;
    font-size: 0.85rem;
}

/* ----------------------------------------------------------------------- */
/* Suppress harmless 404s for fonts that don't exist as files.             */
/* Gradio's default theme tries to fetch /static/fonts/ui-sans-serif/... */
/* and /static/fonts/system-ui/... — those are CSS *keywords*, not real    */
/* font files. Defining empty @font-face rules makes the browser stop      */
/* trying to download them; it just falls back to the actual system font. */
/* ----------------------------------------------------------------------- */
@font-face {
    font-family: "ui-sans-serif";
    src: local("ui-sans-serif"), local("system-ui"), local("sans-serif");
}
@font-face {
    font-family: "system-ui";
    src: local("system-ui"), local("ui-sans-serif"), local("sans-serif");
}
"""


def build_app() -> gr.Blocks:
    cmp_table, cmp_chart, best_name = get_comparison_data()
    best_row = cmp_table[cmp_table["Model"] == best_name]
    if len(best_row) == 0:
        best_row = cmp_table.iloc[[0]]
    best_row = best_row.iloc[0]

    with gr.Blocks(
        title="Network Intrusion Detection - ML Showcase",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        css=CUSTOM_CSS,
    ) as demo:

        gr.HTML(f"""
            <div id="hero">
                <h1>Network Intrusion Detection</h1>
                <p>Real-time scoring of network sessions with a trained machine-learning ensemble.</p>
                <p style="margin-top:14px;">
                    <span class="metric-pill">Model: {MODEL_LABEL}</span>
                    <span class="metric-pill">AUC {best_row['AUC']:.3f}</span>
                    <span class="metric-pill">Recall {best_row['Recall']:.3f}</span>
                    <span class="metric-pill">F1 {best_row['F1-score']:.3f}</span>
                </p>
            </div>
        """)

        with gr.Tabs():

            # ------- TAB 1 -----------
            with gr.Tab("Live prediction"):
                gr.Markdown(
                    "Fill in the parameters of a network session and the "
                    "deployed model will tell you whether it looks malicious."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Network parameters")
                        in_packet = gr.Slider(0, 5000, value=500, step=10,
                                              label="Network packet size (bytes)")
                        in_proto = gr.Radio(["TCP", "UDP", "ICMP"], value="TCP",
                                            label="Protocol type")
                        in_dur = gr.Slider(0, 600, value=120, step=1,
                                           label="Session duration (seconds)")
                        in_enc = gr.Radio(["AES", "DES", "None"], value="AES",
                                          label="Encryption used")
                        in_browser = gr.Dropdown(
                            ["Chrome", "Firefox", "Edge", "Safari", "Unknown"],
                            value="Chrome", label="Browser type",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Authentication & risk signals")
                        in_attempts = gr.Slider(0, 20, value=3, step=1,
                                                label="Login attempts")
                        in_failed = gr.Slider(0, 20, value=2, step=1,
                                              label="Failed logins")
                        in_rep = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                           label="IP reputation score (0 = bad, 1 = good)")
                        in_unusual = gr.Radio(
                            choices=[("No", 0), ("Yes", 1)],
                            value=0,
                            label="Unusual-time access?",
                        )
                        in_threshold = gr.Slider(
                            0.05, 0.95, value=DEFAULT_THRESHOLD, step=0.01,
                            label="Decision threshold (probability of attack)",
                            info="Lower = more sensitive (catches more attacks, more false alarms).",
                        )

                with gr.Row():
                    btn_predict = gr.Button("Run prediction", variant="primary")
                    btn_reset = gr.Button("Reset")

                with gr.Row():
                    out_md = gr.Markdown()
                    out_label = gr.Label(num_top_classes=2, label="Class probabilities")

                gr.Examples(
                    examples=LIVE_EXAMPLES,
                    inputs=[in_packet, in_proto, in_attempts, in_dur, in_enc,
                            in_rep, in_failed, in_browser, in_unusual, in_threshold],
                    label="Try a preset",
                )

                btn_predict.click(
                    predict_session,
                    inputs=[in_packet, in_proto, in_attempts, in_dur, in_enc,
                            in_rep, in_failed, in_browser, in_unusual, in_threshold],
                    outputs=[out_md, out_label],
                    api_name=False,
                )
                btn_reset.click(
                    reset_form,
                    outputs=[in_packet, in_proto, in_attempts, in_dur, in_enc,
                             in_rep, in_failed, in_browser, in_unusual, in_threshold],
                    api_name=False,
                )

            # ------- TAB 2 -----------
            with gr.Tab("Batch prediction"):
                gr.Markdown(
                    "Upload a CSV of sessions to score the whole file at once.\n\n"
                    "**Required columns:** `network_packet_size`, "
                    "`protocol_type`, `login_attempts`, `session_duration`, "
                    "`encryption_used`, `ip_reputation_score`, "
                    "`failed_logins`, `browser_type`, `unusual_time_access`."
                )
                with gr.Row():
                    csv_in = gr.File(label="CSV file", file_types=[".csv"])
                    batch_threshold = gr.Slider(
                        0.05, 0.95, value=DEFAULT_THRESHOLD, step=0.01,
                        label="Decision threshold",
                    )
                btn_batch = gr.Button("Score the file", variant="primary")
                batch_summary = gr.Markdown()
                batch_table = gr.Dataframe(label="Predictions", wrap=True)
                btn_batch.click(
                    predict_batch,
                    inputs=[csv_in, batch_threshold],
                    outputs=[batch_table, batch_summary],
                    api_name=False,
                )

            # ------- TAB 3 -----------
            with gr.Tab("Model comparison"):
                gr.Markdown(
                    f"Every model trained during experimentation, "
                    f"as logged in MLflow. The deployed model is "
                    f"**{best_name}** (selected by mean cross-validation recall - "
                    f"the metric that minimises missed attacks)."
                )
                gr.Dataframe(
                    value=cmp_table,
                    label="Test-set metrics",
                    interactive=False,
                    wrap=True,
                )
                chart_path = build_comparison_chart(cmp_table)
                gr.Image(
                    value=str(chart_path),
                    label="Per-model metric breakdown",
                    show_label=True,
                    show_download_button=False,
                    show_share_button=False,
                    interactive=False,
                    container=True,
                )
                gr.Markdown(
                    "**Reading the table** - *Recall* is what you want to "
                    "maximise here: a missed attack is far worse than a false "
                    "alarm. The deployed ensemble keeps recall high while "
                    "preserving a competitive precision."
                )

            # ------- TAB 4 -----------
            with gr.Tab("EDA & explainability"):
                eda_items = _gallery_items(EDA_PLOTS)
                eval_items = _gallery_items(EVAL_PLOTS)

                gr.Markdown("### Exploratory data analysis")
                if eda_items:
                    gr.Gallery(
                        value=eda_items,
                        columns=2,
                        height="auto",
                        object_fit="contain",
                        show_label=False,
                        allow_preview=True,
                    )
                else:
                    gr.Markdown("_No EDA plot found in `outputs/`._")

                gr.Markdown("### Model evaluation & SHAP")
                if eval_items:
                    gr.Gallery(
                        value=eval_items,
                        columns=2,
                        height="auto",
                        object_fit="contain",
                        show_label=False,
                        allow_preview=True,
                    )
                else:
                    gr.Markdown(
                        "_No evaluation plot found yet - run "
                        "`python src/evaluate.py` and `python src/explain.py` "
                        "to generate them._"
                    )

            # ------- TAB 5 -----------
            with gr.Tab("About"):
                gr.Markdown(f"""
### About this project

Binary classifier that flags malicious network sessions from behavioural
signals (packet size, login behaviour, IP reputation, encryption, etc.).

**Pipeline**
1. Preprocessing - clean, impute, one-hot encode, add engineered features
   (`risk_index`, `login_speed`).
2. Training - five candidate models with class weighting and 5-fold
   stratified cross-validation; threshold optimised for >= 80% recall.
3. Selection - the model with the highest mean cross-validation recall is
   serialised to `models/model.pkl`.
4. Tracking - every run logged to MLflow (`mlflow ui` to browse).
5. Serving - this Gradio frontend (and a sibling Flask REST API in
   `src/app.py`).

**Currently deployed:** `{MODEL_LABEL}` - AUC {best_row['AUC']:.3f},
recall {best_row['Recall']:.3f}, F1 {best_row['F1-score']:.3f}.

### Feature reference
| Feature | Type | Notes |
|---|---|---|
| `network_packet_size` | numerical | bytes |
| `protocol_type` | categorical | TCP / UDP / ICMP |
| `login_attempts` | numerical | total attempts in the session |
| `session_duration` | numerical | seconds |
| `encryption_used` | categorical | AES / DES / None |
| `ip_reputation_score` | numerical | 0 = malicious, 1 = trusted |
| `failed_logins` | numerical | failed authentication attempts |
| `browser_type` | categorical | Chrome / Firefox / Edge / Safari / Unknown |
| `unusual_time_access` | binary | accessed outside business hours? |

### Engineered features (added automatically)
- `risk_index = (1 - ip_reputation_score) * failed_logins`
- `login_speed = login_attempts / (session_duration + 0.1)`

### What's next
The next milestone is containerisation (Docker) so this app can be served
anywhere with a single command.
""")

    return demo


# --- Entry point ------------------------------------------------------------

def _print_startup_diagnostics() -> None:
    """Tell the user up-front what the dynamic tabs found on disk.

    Saves a lot of time when the comparison / EDA tabs come up empty:
    you immediately see whether MLflow + outputs/ are reachable.
    """
    print(f"[gradio] Loading deployed model : {MODEL_LABEL}")
    print(f"[gradio] Expected feature count : {len(EXPECTED_COLS)}")
    print(f"[gradio] Repo root              : {ROOT}")

    # MLflow data
    runs = _read_runs_from_mlflow()
    if runs:
        print(f"[gradio] MLflow runs picked up  : {len(runs)} model(s) "
              f"({', '.join(name for name, *_ in runs)})")
    else:
        print("[gradio] MLflow runs picked up  : 0 (using FALLBACK_RUNS)")

    # Plots
    eda  = [fn for fn, _ in EDA_PLOTS  if (OUTPUTS_DIR / fn).exists()]
    eval_ = [fn for fn, _ in EVAL_PLOTS if (OUTPUTS_DIR / fn).exists()]
    print(f"[gradio] EDA plots found        : {len(eda)}/{len(EDA_PLOTS)} "
          f"({', '.join(eda) if eda else 'none'})")
    print(f"[gradio] Evaluation/SHAP plots  : {len(eval_)}/{len(EVAL_PLOTS)} "
          f"({', '.join(eval_) if eval_ else 'none'})")
    print(f"[gradio] Comparison chart path  : {COMPARISON_CHART_PATH} "
          f"(will be (re)generated on app build)")
    if not eda or not eval_:
        print("[gradio] Tip: run `python src/eda.py`, `python src/evaluate.py` "
              "and `python src/explain.py` to (re)generate the plots.")


if __name__ == "__main__":
    # Port is configurable via env var so the same image works on
    # Azure Container Apps / App Service (PORT) and locally (default 7860).
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")

    _print_startup_diagnostics()
    print(f"[gradio] Listening on http://{host}:{port}")

    app = build_app()
    # allowed_paths is required by Gradio >= 5 to serve files that live
    # outside the cwd (the Gallery on tab 4 reads PNGs from outputs/).
    app.launch(
        server_name=host,
        server_port=port,
        show_error=True,
        allowed_paths=[str(OUTPUTS_DIR)],
    )