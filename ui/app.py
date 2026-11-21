"""Interactive Streamlit dashboard for Code Plagiarism Detection.

Communicates strictly with the FastAPI backend over HTTP via httpx and never
imports from src/.
"""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import os
from typing import Any

import httpx
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

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

DIFF2HTML_CSS_URL = (
    "https://cdn.jsdelivr.net/npm/diff2html@3.4.51/bundles/css/diff2html.min.css"
)
DIFF2HTML_JS_URL = (
    "https://cdn.jsdelivr.net/npm/diff2html@3.4.51/bundles/js/diff2html-ui.min.js"
)
HLJS_CSS_URL = (
    "https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css"
)

BASELINE_METRIC_SPECS: tuple[dict[str, str], ...] = (
    {
        "key": "markov_similarity_pct",
        "label": "Markov Similarity",
        "format": "pct",
        "description": (
            "Percentage match between AST transition patterns. "
            "[More](https://en.wikipedia.org/wiki/Markov_chain)"
        ),
        "tooltip": (
            "An abstract syntax tree (AST) maps code into structural elements "
            "such as loops and function calls. A Markov chain counts how often "
            "one structural element follows another. This score converts the "
            "transition distance between Code A and Code B into a percentage.\n\n"
            "Values near 100% mean both files follow nearly identical structural "
            "flow. Values near 0% mean their control structures follow very "
            "different patterns."
        ),
    },
    {
        "key": "tfidf_cosine_score",
        "label": "TF IDF Cosine",
        "format": "float4",
        "description": (
            "Text similarity across overlapping character sequences. "
            "[More](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)"
        ),
        "tooltip": (
            "Term frequency inverse document frequency (TF IDF) weights short "
            "character sequences by how rare they are across programs. Cosine "
            "similarity measures the angle between the weighted text vectors "
            "of Code A and Code B.\n\n"
            "Values near 1.0 mean both files share almost identical wording and "
            "tokens. Values near 0.0 mean the raw text has little lexical overlap."
        ),
    },
    {
        "key": "markov_jsd",
        "label": "Markov JSD",
        "format": "float4",
        "description": (
            "Jensen Shannon divergence between transition probability matrices. "
            "[More](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)"
        ),
        "tooltip": (
            "A transition probability matrix records the likelihood of moving "
            "from one AST state to the next. Jensen Shannon divergence (JSD) is "
            "a symmetric statistical distance that compares two probability "
            "distributions.\n\n"
            "Values near 0.0 mean both files share nearly identical state "
            "transition probabilities. Higher values mean their structural "
            "transition patterns diverge."
        ),
    },
    {
        "key": "markov_frobenius",
        "label": "Markov Frobenius",
        "format": "float4",
        "description": (
            "Euclidean distance between transition probability matrices. "
            "[More](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm)"
        ),
        "tooltip": (
            "A transition probability matrix stores how often each AST state "
            "moves to another state. The Frobenius norm sums the squared "
            "differences across every cell in |P_A - P_B| and takes the square "
            "root.\n\n"
            "Values near 0.0 mean every transition probability cell matches "
            "closely between both files. Higher values mean larger cell level "
            "differences across the matrices."
        ),
    },
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
    code_a: str,
    code_b: str,
    language: str,
    operating_mode: str,
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """Send POST /predict to the FastAPI backend and classify errors."""
    payload = {
        "code_a": code_a,
        "code_b": code_b,
        "language": language,
        "operating_mode": operating_mode,
    }
    try:
        response = httpx.post(
            f"{api_url}/predict",
            json=payload,
            timeout=PREDICT_TIMEOUT_S,
        )
    except httpx.TimeoutException:
        return (
            None,
            None,
            f"Analysis timed out after {PREDICT_TIMEOUT_S:.0f}s.",
        )
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

    detail = _extract_error_detail(response)
    return None, response.status_code, detail


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
    transition_delta: list[list[float]],
) -> plt.Figure:
    """Render a 12x12 AST transition delta heatmap |P_A - P_B|."""
    size = len(state_names)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(transition_delta, cmap="YlOrRd", aspect="auto", vmin=0.0)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(state_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(state_names, fontsize=8)
    ax.set_xlabel("Next AST Structural State")
    ax.set_ylabel("Current AST Structural State")
    ax.set_title("AST Transition Delta Heatmap (|P_A - P_B|)")

    max_val = max(
        (val for row in transition_delta for val in row),
        default=0.0,
    )
    threshold = max_val * 0.55 if max_val > 0 else 0.5

    for row_idx, row in enumerate(transition_delta):
        for col_idx, val in enumerate(row):
            if val >= 0.01:
                text_color = "white" if val >= threshold else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=6.5,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("|P_A - P_B|")
    fig.tight_layout()
    return fig


def build_unified_diff(code_a: str, code_b: str, language: str = "java") -> str:
    """Return a unified diff comparing Code A and Code B with language extension."""
    ext_map = {"java": "java", "python": "py", "cpp": "cpp"}
    ext = ext_map.get(language, "txt")
    diff_lines = difflib.unified_diff(
        code_a.splitlines(keepends=True),
        code_b.splitlines(keepends=True),
        fromfile=f"Code_A.{ext}",
        tofile=f"Code_B.{ext}",
    )
    return "".join(diff_lines)


def build_diff2html_document(unified_diff: str) -> str:
    """Build a standalone HTML document rendering a side by side diff2html view."""
    diff_js_literal = json.dumps(unified_diff).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="{HLJS_CSS_URL}" />
  <link rel="stylesheet" href="{DIFF2HTML_CSS_URL}" />
  <script src="{DIFF2HTML_JS_URL}"></script>
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: transparent;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    .d2h-wrapper {{
      margin: 0;
    }}
  </style>
</head>
<body>
  <div id="diff-target"></div>
  <script>
    const diffString = {diff_js_literal};
    const targetElement = document.getElementById("diff-target");
    const configuration = {{
      drawFileList: false,
      fileContentToggle: false,
      matching: "words",
      outputFormat: "side-by-side",
      synchronisedScroll: true,
      highlight: true,
      renderNothingWhenEmpty: false
    }};
    const diff2htmlUi = new Diff2HtmlUI(targetElement, diffString, configuration);
    diff2htmlUi.draw();
    diff2htmlUi.highlightCode();
  </script>
</body>
</html>
"""


def _init_session_state() -> None:
    """Initialize session state keys on first page load."""
    if "source_selector" not in st.session_state:
        st.session_state["source_selector"] = UPLOAD_SOURCE_LABEL
    if "code_a" not in st.session_state:
        st.session_state["code_a"] = ""
    if "code_b" not in st.session_state:
        st.session_state["code_b"] = ""
    if "baseline_hash_a" not in st.session_state:
        st.session_state["baseline_hash_a"] = _content_hash("")
    if "baseline_hash_b" not in st.session_state:
        st.session_state["baseline_hash_b"] = _content_hash("")
    if "last_upload_a" not in st.session_state:
        st.session_state["last_upload_a"] = None
    if "last_upload_b" not in st.session_state:
        st.session_state["last_upload_b"] = None
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None
    if "last_analysis_key" not in st.session_state:
        st.session_state["last_analysis_key"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "code_inputs_expanded" not in st.session_state:
        st.session_state["code_inputs_expanded"] = False


def _on_source_change() -> None:
    """Sync Code A and Code B when the user changes the Source selector."""
    selected = st.session_state["source_selector"]
    st.session_state["code_inputs_expanded"] = selected == MANUAL_SOURCE_LABEL
    if selected in DEMO_SAMPLES:
        sample = DEMO_SAMPLES[selected]
        st.session_state["code_a"] = sample["code_a"]
        st.session_state["code_b"] = sample["code_b"]
        st.session_state["baseline_hash_a"] = _content_hash(sample["code_a"])
        st.session_state["baseline_hash_b"] = _content_hash(sample["code_b"])
    elif selected == UPLOAD_SOURCE_LABEL:
        # Preserve existing editor text until a new file is uploaded.
        st.session_state["baseline_hash_a"] = _content_hash(st.session_state["code_a"])
        st.session_state["baseline_hash_b"] = _content_hash(st.session_state["code_b"])
    else:
        st.session_state["baseline_hash_a"] = None
        st.session_state["baseline_hash_b"] = None


def _on_code_edit() -> None:
    """Switch Source to Manual Input when edited text diverges from loaded content."""
    st.session_state["code_inputs_expanded"] = True
    if st.session_state["source_selector"] == MANUAL_SOURCE_LABEL:
        return

    current_hash_a = _content_hash(st.session_state["code_a"])
    current_hash_b = _content_hash(st.session_state["code_b"])
    if (
        current_hash_a != st.session_state["baseline_hash_a"]
        or current_hash_b != st.session_state["baseline_hash_b"]
    ):
        st.session_state["source_selector"] = MANUAL_SOURCE_LABEL
        st.session_state["baseline_hash_a"] = None
        st.session_state["baseline_hash_b"] = None


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


def _render_sidebar() -> str:
    """Render sidebar API status, controls, and conditional file uploaders."""
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

        language_options = list(dict.fromkeys(["java", *supported_languages]))
        language = st.selectbox(
            "Language",
            options=language_options,
            index=0,
            format_func=lambda lang: lang.upper() if lang == "cpp" else lang.title(),
            help="The model is calibrated for Java.",
        )

        source_options = [
            UPLOAD_SOURCE_LABEL,
            MANUAL_SOURCE_LABEL,
            *DEMO_SAMPLES.keys(),
        ]
        selected_source = st.selectbox(
            "Source",
            options=source_options,
            key="source_selector",
            on_change=_on_source_change,
        )

        if selected_source == UPLOAD_SOURCE_LABEL:
            uploaded_a = st.file_uploader(
                "Upload Code A",
                type=["java"],
                key="uploader_a",
            )
            if uploaded_a is not None:
                upload_token_a = f"{uploaded_a.name}:{uploaded_a.size}"
                if upload_token_a != st.session_state["last_upload_a"]:
                    decoded_a = uploaded_a.getvalue().decode("utf-8", errors="replace")
                    st.session_state["code_a"] = decoded_a
                    st.session_state["baseline_hash_a"] = _content_hash(decoded_a)
                    st.session_state["last_upload_a"] = upload_token_a

            uploaded_b = st.file_uploader(
                "Upload Code B",
                type=["java"],
                key="uploader_b",
            )
            if uploaded_b is not None:
                upload_token_b = f"{uploaded_b.name}:{uploaded_b.size}"
                if upload_token_b != st.session_state["last_upload_b"]:
                    decoded_b = uploaded_b.getvalue().decode("utf-8", errors="replace")
                    st.session_state["code_b"] = decoded_b
                    st.session_state["baseline_hash_b"] = _content_hash(decoded_b)
                    st.session_state["last_upload_b"] = upload_token_b

    return language


def _render_verdict_kpi(prob: float, predicted_label: int) -> None:
    """Render Verdict and Plagiarism Probability as native KPI metrics with colored values."""
    is_plag = predicted_label == 1
    verdict_text = "Plagiarized" if is_plag else "Clean"
    container_key = "primary_kpi_plag" if is_plag else "primary_kpi_clean"

    st.markdown(
        """
        <style>
          .st-key-primary_kpi_plag [data-testid="stMetricValue"] {
            color: #dc2626;
            font-weight: 700;
          }
          .st-key-primary_kpi_clean [data-testid="stMetricValue"] {
            color: #16a34a;
            font-weight: 700;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key=container_key):
        kpi_col_a, kpi_col_b = st.columns(2, gap="medium")
        kpi_col_a.metric("Verdict", verdict_text)
        kpi_col_b.metric("Plagiarism Probability", f"{prob * 100:.1f}%")


def _format_metric_value(raw_value: float, fmt: str) -> str:
    """Format a baseline metric value for display."""
    if fmt == "pct":
        return f"{raw_value:.1f}%"
    return f"{raw_value:.4f}"


def _render_inline_hint(message: str, tone: str = "amber") -> None:
    """Render a compact colored validation message directly under an input."""
    color = "#dc2626" if tone == "red" else "#b45309"
    safe_msg = html.escape(message)
    st.markdown(
        f'<div style="font-size:0.85rem;color:{color};margin-top:-0.5rem;margin-bottom:0.5rem;">'
        f"{safe_msg}</div>",
        unsafe_allow_html=True,
    )


def _render_results(
    result: dict[str, Any],
    code_a: str,
    code_b: str,
    language: str,
) -> None:
    """Render unified Analysis Results and Side by Side Code Diff."""
    st.divider()
    st.subheader("Analysis Results")

    prob = float(result.get("plagiarism_probability", 0.0))
    predicted_label = int(result.get("predicted_label", 0))
    state_names = list(result.get("state_names", []))
    transition_delta = list(result.get("transition_delta", []))

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        _render_verdict_kpi(prob=prob, predicted_label=predicted_label)

        for row_start in range(0, len(BASELINE_METRIC_SPECS), 2):
            row_cols = st.columns(2, gap="medium")
            for col, spec in zip(
                row_cols,
                BASELINE_METRIC_SPECS[row_start : row_start + 2],
            ):
                val = float(result.get(spec["key"], 0.0))
                with col:
                    st.metric(
                        label=spec["label"],
                        value=_format_metric_value(val, spec["format"]),
                        help=spec["tooltip"],
                    )
                    st.caption(spec["description"])

    with right_col:
        if (
            state_names
            and transition_delta
            and len(transition_delta) == len(state_names)
            and all(len(row) == len(state_names) for row in transition_delta)
        ):
            fig = render_transition_heatmap(state_names, transition_delta)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.warning(
                "Transition matrix data was empty or mismatched. "
                "The heatmap did not render."
            )

    st.divider()
    st.subheader("Code Diff")
    unified_text = build_unified_diff(code_a, code_b, language=language)
    if unified_text:
        diff_html = build_diff2html_document(unified_text)
        line_count = max(len(code_a.splitlines()), len(code_b.splitlines()), 10)
        frame_height = min(max(line_count * 24 + 90, 320), 760)
        components.html(diff_html, height=frame_height, scrolling=True)
    else:
        st.info("Code A and Code B are identical.")


def _run_analysis_if_ready(
    code_a: str,
    code_b: str,
    language: str,
) -> None:
    """Auto run prediction when inputs change and both editors are non empty."""
    if not code_a.strip() or not code_b.strip():
        st.session_state["last_result"] = None
        st.session_state["last_analysis_key"] = None
        st.session_state["last_error"] = None
        return

    analysis_key = (
        _content_hash(code_a),
        _content_hash(code_b),
        language,
    )
    if analysis_key == st.session_state["last_analysis_key"]:
        return

    with st.spinner("Running analysis..."):
        result, status_code, error_detail = request_prediction(
            api_url=API_URL,
            code_a=code_a,
            code_b=code_b,
            language=language,
            operating_mode="balanced",
        )

    st.session_state["last_analysis_key"] = analysis_key
    if result is not None:
        st.session_state["last_result"] = {
            "result": result,
            "code_a": code_a,
            "code_b": code_b,
            "language": language,
        }
        st.session_state["last_error"] = None
    else:
        st.session_state["last_result"] = None
        st.session_state["last_error"] = (status_code, error_detail)


def main() -> None:
    """Entrypoint for the Streamlit application."""
    st.set_page_config(
        page_title="Code Plagiarism Detector",
        layout="wide",
    )
    _init_session_state()

    language = _render_sidebar()

    st.title("Code Plagiarism Detector")

    code_inputs_expander = st.expander(
        "Code Inputs",
        expanded=False,
        key="code_inputs_expanded",
        on_change="rerun",
    )
    with code_inputs_expander:
        editor_col_a, editor_col_b = st.columns(2)
        with editor_col_a:
            code_a = st.text_area(
                "Code A",
                key="code_a",
                height=340,
                on_change=_on_code_edit,
            )
            hint_slot_a = st.empty()
        with editor_col_b:
            code_b = st.text_area(
                "Code B",
                key="code_b",
                height=340,
                on_change=_on_code_edit,
            )
            hint_slot_b = st.empty()

    _run_analysis_if_ready(
        code_a=code_a,
        code_b=code_b,
        language=language,
    )

    has_a = bool(code_a.strip())
    has_b = bool(code_b.strip())
    saved = st.session_state.get("last_result")
    last_error = st.session_state.get("last_error")

    hint_a: str | None = None
    if has_b and not has_a:
        hint_a = "Add code to Code A to run the comparison."
    elif saved is not None and bool(saved["result"].get("parse_warnings_a", False)):
        hint_a = "Syntax error found. Partial tree used."

    hint_b: str | None = None
    if has_a and not has_b:
        hint_b = "Add code to Code B to run the comparison."
    elif saved is not None and bool(saved["result"].get("parse_warnings_b", False)):
        hint_b = "Syntax error found. Partial tree used."

    if code_inputs_expander.open:
        if hint_a is not None:
            with hint_slot_a:
                _render_inline_hint(hint_a, tone="amber")
        if hint_b is not None:
            with hint_slot_b:
                _render_inline_hint(hint_b, tone="amber")
    else:
        if hint_a is not None:
            _render_inline_hint(f"Code A: {hint_a}", tone="amber")
        if hint_b is not None:
            _render_inline_hint(f"Code B: {hint_b}", tone="amber")

    if last_error is not None:
        status_code, error_detail = last_error
        if status_code == 422:
            _render_inline_hint(str(error_detail), tone="red")
        elif status_code == 503:
            st.error(
                f"Model service is not ready (HTTP 503): {error_detail}. "
                "Make sure that the backend loaded the model files."
            )
        elif status_code is not None:
            st.error(f"Backend failed with HTTP {status_code}: {error_detail}")
        else:
            st.error(f"Connection to {API_URL} failed: {error_detail}")

    if saved is not None:
        _render_results(
            result=saved["result"],
            code_a=saved["code_a"],
            code_b=saved["code_b"],
            language=saved.get("language", "java"),
        )


if __name__ == "__main__":
    main()
