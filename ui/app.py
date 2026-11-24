"""Interactive Streamlit dashboard for the AI Code Detector.

Communicates strictly with the FastAPI backend over HTTP via httpx and never
imports from src/.
"""

from __future__ import annotations

import hashlib
import html
import io
import os
from typing import Any

import httpx
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

matplotlib.use("Agg")

try:
    from ui.samples import DEMO_SAMPLES
except ImportError:
    from samples import DEMO_SAMPLES

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
HEALTH_TIMEOUT_S = 2.5
PREDICT_TIMEOUT_S = 30.0
UPLOAD_SOURCE_LABEL = "Upload Input"
MANUAL_SOURCE_LABEL = "Manual Input"

MODE_LABELS = {"balanced": "Balanced", "low_fpr": "Fewer false alarms"}
MODE_HELP = (
    "Balanced: the default threshold, which catches more AI code but also flags more human code. "
    "Fewer false alarms: a higher threshold, which flags less human code but misses more AI code."
)
DISCLAIMER = (
    "This is a statistical signal based only on the structure of the code (comments and formatting are ignored). "
    "It is moderately accurate: some human functions are flagged as AI and some AI functions are missed. "
    "Do not use it as proof of authorship."
)


def _content_hash(text: str) -> str:
    """Return a deterministic SHA 256 digest for editor content comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_backend_health(api_url: str) -> tuple[dict[str, Any] | None, str | None]:
    """Query GET /health on the FastAPI backend."""
    try:
        response = httpx.get(f"{api_url}/health", timeout=HEALTH_TIMEOUT_S)
    except httpx.TimeoutException:
        return None, f"Health request timed out after {HEALTH_TIMEOUT_S:.0f}s."
    except httpx.RequestError as exc:
        return None, f"Connection to {api_url} failed: {exc}"

    if response.status_code != 200:
        return None, f"Health request failed with HTTP {response.status_code}."

    try:
        payload = response.json()
    except ValueError:
        return None, "Health response was not valid JSON."

    if not isinstance(payload, dict):
        return None, "Health response format was not valid."

    return payload, None


def request_prediction(
    api_url: str,
    code: str,
    language: str,
    operating_mode: str,
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """Send POST /predict to the FastAPI backend and classify errors."""
    payload = {"code": code, "language": language, "operating_mode": operating_mode}
    try:
        response = httpx.post(f"{api_url}/predict", json=payload, timeout=PREDICT_TIMEOUT_S)
    except httpx.TimeoutException:
        return None, None, f"Analysis timed out after {PREDICT_TIMEOUT_S:.0f}s."
    except httpx.RequestError as exc:
        return None, None, f"Connection to {api_url}/predict failed: {exc}"

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            return None, 200, "Prediction response was not valid JSON."
        if not isinstance(data, dict):
            return None, 200, "Prediction response format was not valid."
        return data, 200, None

    return None, response.status_code, _extract_error_detail(response)


def _extract_error_detail(response: httpx.Response) -> str:
    """Extract a readable error detail string from an HTTP error response."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or f"HTTP {response.status_code}"

    if isinstance(body, dict) and "detail" in body:
        detail = body["detail"]
        if isinstance(detail, list):
            parts = []
            for item in detail:
                if isinstance(item, dict):
                    loc = ".".join(str(x) for x in item.get("loc", []))
                    msg = item.get("msg", str(item))
                    parts.append(f"{loc}: {msg}" if loc else str(msg))
                else:
                    parts.append(str(item))
            return "; ".join(parts)
        return str(detail)

    return str(body)


def render_transition_heatmap(
    state_names: list[str],
    transition_matrix: list[list[float]],
) -> plt.Figure:
    """Render the 12x12 AST transition matrix of the analyzed function."""
    size = len(state_names)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(transition_matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(state_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(state_names, fontsize=8)
    ax.set_xlabel("Next AST Structural State")
    ax.set_ylabel("Current AST Structural State")
    ax.set_title("AST Transition Matrix of Your Function")

    for row_idx, row in enumerate(transition_matrix):
        for col_idx, val in enumerate(row):
            if val >= 0.01:
                ax.text(
                    col_idx,
                    row_idx,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val >= 0.55 else "black",
                    fontsize=6.5,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(next state | current state)")
    fig.tight_layout()
    return fig


def _init_session_state() -> None:
    """Initialize session state keys on first page load."""
    defaults: dict[str, Any] = {
        "source_selector": MANUAL_SOURCE_LABEL,
        "code": "",
        "baseline_hash": _content_hash(""),
        "last_upload": None,
        "last_result": None,
        "last_analysis_key": None,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _on_source_change() -> None:
    """Load the selected demo sample into the editor when the Source selector changes."""
    selected = st.session_state["source_selector"]
    if selected in DEMO_SAMPLES:
        sample_code = DEMO_SAMPLES[selected]["code"]
        st.session_state["code"] = sample_code
        st.session_state["baseline_hash"] = _content_hash(sample_code)
    elif selected == UPLOAD_SOURCE_LABEL:
        st.session_state["baseline_hash"] = _content_hash(st.session_state["code"])
    else:
        st.session_state["baseline_hash"] = None


def _on_code_edit() -> None:
    """Switch Source to Manual Input when edited text diverges from loaded content."""
    if st.session_state["source_selector"] == MANUAL_SOURCE_LABEL:
        return
    if _content_hash(st.session_state["code"]) != st.session_state["baseline_hash"]:
        st.session_state["source_selector"] = MANUAL_SOURCE_LABEL
        st.session_state["baseline_hash"] = None


def _classify_health(
    health: dict[str, Any] | None,
    health_error: str | None,
) -> tuple[str, str, bool]:
    """Return (dot_hex_color, tooltip_text, is_green) for the API status dot."""
    if health is not None:
        status_val = str(health.get("status", "unknown"))
        model_loaded = bool(health.get("model_loaded", False))
        model_version = str(health.get("model_version", "unknown"))
        if status_val == "ok" and model_loaded:
            return "#22c55e", f"API ready (v{model_version})", True
        if status_val == "ok":
            return "#f59e0b", "API reached. Model did not load.", False
        return "#f59e0b", f"API status: {status_val}", False

    tooltip = health_error or "API did not respond."
    return "#ef4444", tooltip, False


def _render_api_status_row(color: str, tooltip: str) -> None:
    """Render API Status heading with a right aligned status dot on the same row."""
    safe_tooltip = html.escape(tooltip, quote=True)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.5rem;">
          <span style="font-size:1.25rem;font-weight:600;line-height:1.2;">API Status</span>
          <span title="{safe_tooltip}" style="
            display:inline-block;
            width:12px;
            height:12px;
            border-radius:50%;
            background-color:{color};
            box-shadow:0 0 0 2px rgba(0,0,0,0.08);
            cursor:help;
          "></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=3)
def _poll_api_status_until_ready() -> None:
    """Auto refresh API status every 3 seconds while the backend is not ready."""
    health, health_error = fetch_backend_health(API_URL)
    color, tooltip, is_green = _classify_health(health, health_error)
    _render_api_status_row(color, tooltip)
    if is_green:
        st.rerun()


def _render_sidebar() -> tuple[str, str]:
    """Render sidebar API status and controls; return (language, operating_mode)."""
    health, health_error = fetch_backend_health(API_URL)
    color, tooltip, is_green = _classify_health(health, health_error)
    supported_languages = (
        health.get("supported_languages") if isinstance(health, dict) else None
    ) or ["java"]

    with st.sidebar:
        if is_green:
            _render_api_status_row(color, tooltip)
        else:
            _poll_api_status_until_ready()

        st.divider()
        st.header("Controls")

        language = st.selectbox(
            "Language",
            options=list(supported_languages),
            index=0,
            format_func=lambda lang: lang.title(),
            help="The model is calibrated for Java methods.",
        )

        operating_mode = st.radio(
            "Sensitivity",
            options=list(MODE_LABELS),
            format_func=lambda mode: MODE_LABELS[mode],
            help=MODE_HELP,
        )

        selected_source = st.selectbox(
            "Source",
            options=[MANUAL_SOURCE_LABEL, UPLOAD_SOURCE_LABEL, *DEMO_SAMPLES.keys()],
            key="source_selector",
            on_change=_on_source_change,
        )

        if selected_source == UPLOAD_SOURCE_LABEL:
            uploaded = st.file_uploader("Upload a Java file", type=["java"], key="uploader")
            if uploaded is not None:
                token = f"{uploaded.name}:{uploaded.size}"
                if token != st.session_state["last_upload"]:
                    decoded = uploaded.getvalue().decode("utf-8", errors="replace")
                    st.session_state["code"] = decoded
                    st.session_state["baseline_hash"] = _content_hash(decoded)
                    st.session_state["last_upload"] = token

        if selected_source in DEMO_SAMPLES:
            sample = DEMO_SAMPLES[selected_source]
            st.caption(sample["description"])
            st.caption(f"Actually written by: **{sample['expected_label']}**")

    return language, operating_mode


def _render_verdict(result: dict[str, Any]) -> None:
    """Render the verdict and AI probability as native KPI metrics with colored values."""
    prob = float(result.get("ai_probability", 0.0))
    is_ai = int(result.get("predicted_label", 0)) == 1
    container_key = "kpi_ai" if is_ai else "kpi_human"

    st.markdown(
        """
        <style>
          .st-key-kpi_ai [data-testid="stMetricValue"] { color: #d97706; font-weight: 700; }
          .st-key-kpi_human [data-testid="stMetricValue"] { color: #2563eb; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key=container_key):
        col_a, col_b = st.columns(2, gap="medium")
        col_a.metric("Verdict", "Likely AI-written" if is_ai else "Likely human-written")
        col_b.metric("AI Probability", f"{prob * 100:.1f}%")

    st.progress(min(max(prob, 0.0), 1.0))
    mode = MODE_LABELS.get(str(result.get("operating_mode", "balanced")), "Balanced")
    st.caption(
        f"Flagged as AI when the probability reaches {float(result.get('threshold_used', 0.5)) * 100:.0f}% "
        f"(sensitivity: {mode})."
    )


def _render_details(result: dict[str, Any]) -> None:
    """Render the structural details of the analyzed function."""
    col_a, col_b = st.columns(2, gap="medium")
    col_a.metric(
        "AST Structural Tokens",
        int(result.get("ast_tokens", 0)),
        help="Number of structural elements (loops, calls, assignments...) found in the code.",
    )
    col_b.metric(
        "Syntax Check",
        "Partial tree" if result.get("parse_warnings") else "Clean",
        help="If the code has syntax errors, the model still uses the part of the tree it could parse.",
    )
    note = result.get("input_note")
    if note:
        st.warning(str(note))


def _heatmap_png(result: dict[str, Any]) -> bytes | None:
    """Render the heatmap of the analyzed function to PNG bytes."""
    state_names = result.get("state_names")
    matrix = result.get("transition_matrix")
    if not isinstance(state_names, list) or not isinstance(matrix, list):
        return None
    fig = render_transition_heatmap(state_names, matrix)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def _analyze(code: str, language: str, operating_mode: str) -> None:
    """Call the API unless this exact input was already analyzed, and store the outcome in session state."""
    key = _content_hash(f"{language}|{operating_mode}|{code}")
    if key == st.session_state["last_analysis_key"]:
        return
    with st.spinner("Analyzing..."):
        result, status, error = request_prediction(API_URL, code, language, operating_mode)
    st.session_state["last_analysis_key"] = key
    if result is not None:
        st.session_state["last_result"] = result
        st.session_state["last_error"] = None
    else:
        st.session_state["last_result"] = None
        st.session_state["last_error"] = (status, error)


def main() -> None:
    """Entrypoint for the Streamlit application."""
    st.set_page_config(page_title="AI Code Detector", layout="wide")
    _init_session_state()

    language, operating_mode = _render_sidebar()

    st.title("AI Code Detector")
    st.write(
        "Paste one Java function and the model estimates whether it was written by a human or by an AI, "
        "based on the structure of the code."
    )
    code = st.text_area("Java function", key="code", height=320, on_change=_on_code_edit)

    st.divider()
    st.subheader("Analysis Results")

    if not code.strip():
        st.session_state["last_analysis_key"] = None
        st.session_state["last_result"] = None
        st.session_state["last_error"] = None
        st.info("Paste a Java function above, upload a file, or pick a sample in the sidebar.")
    else:
        _analyze(code, language, operating_mode)

    result = st.session_state["last_result"]
    last_error = st.session_state["last_error"]

    if result is not None:
        left, right = st.columns(2, gap="large")
        with left:
            _render_verdict(result)
            _render_details(result)
            st.caption(DISCLAIMER)
        with right:
            png = _heatmap_png(result)
            if png is not None:
                st.image(png)
    elif last_error is not None:
        status_code, detail = last_error
        if status_code == 422:
            st.warning(str(detail))
        elif status_code == 503:
            st.error(
                f"Model service is not ready (HTTP 503): {detail}. "
                "Make sure that the backend loaded the model files."
            )
        elif status_code is not None:
            st.error(f"Backend failed with HTTP {status_code}: {detail}")
        else:
            st.error(f"Connection to {API_URL} failed: {detail}")


if __name__ == "__main__":
    main()
