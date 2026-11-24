"""Interactive Streamlit dashboard for Code Plagiarism Detection.

Communicates strictly with the FastAPI backend over HTTP via httpx and never
imports from src/.
"""

from __future__ import annotations

import difflib
import hashlib
import html
import io
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
MODE_PAIRWISE_PLAGIARISM = "Pairwise Plagiarism"
MODE_AI_PLAGIARISM = "AI Plagiarism"
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


AI_BASELINE_METRIC_SPECS: tuple[dict[str, str], ...] = (
    {
        "key": "perplexity",
        "label": "AST Perplexity",
        "format": "float4",
        "description": (
            "Predictability of AST structural state transitions. "
            "[More](https://en.wikipedia.org/wiki/Perplexity)"
        ),
        "tooltip": (
            "Transition perplexity measures how predictable each next AST state "
            "is under the file's Markov transition matrix.\n\n"
            "Lower values mean the code follows repetitive, highly predictable "
            "structural transitions. Higher values mean more varied control flow."
        ),
    },
    {
        "key": "ast_tokens",
        "label": "AST Tokens",
        "format": "int",
        "description": "Mapped structural syntax nodes extracted from Code.",
        "tooltip": (
            "Total number of abstract syntax tree (AST) nodes mapped into the "
            "12 structural states for Markov transition modeling."
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


def _post_predict_endpoint(
    url: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """Send a JSON POST prediction request to the FastAPI backend and classify errors."""
    try:
        response = httpx.post(url, json=payload, timeout=PREDICT_TIMEOUT_S)
    except httpx.TimeoutException:
        return None, None, f"Analysis timed out after {PREDICT_TIMEOUT_S:.0f}s."
    except httpx.RequestError as exc:
        return None, None, f"Connection to {url} failed: {exc}"

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


def request_prediction(
    api_url: str,
    code_a: str,
    code_b: str,
    language: str,
    operating_mode: str = "balanced",
    model: str | None = None,
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """Send POST /predict/pairwise_plagiarism to the FastAPI backend."""
    payload: dict[str, Any] = {
        "code_a": code_a,
        "code_b": code_b,
        "language": language,
        "operating_mode": operating_mode,
    }
    if model is not None:
        payload["model"] = model
    return _post_predict_endpoint(f"{api_url}/predict/pairwise_plagiarism", payload)


def request_ai_plagiarism_prediction(
    api_url: str,
    code: str,
    language: str,
    operating_mode: str = "balanced",
    model: str | None = None,
) -> tuple[dict[str, Any] | None, int | None, str | None]:
    """Send single file POST /predict/ai_plagiarism to the FastAPI backend."""
    payload: dict[str, Any] = {
        "code": code,
        "language": language,
        "operating_mode": operating_mode,
    }
    if model is not None:
        payload["model"] = model
    return _post_predict_endpoint(f"{api_url}/predict/ai_plagiarism", payload)


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
    title: str = "AST Transition Delta Heatmap (|P_A - P_B|)",
    colorbar_label: str = "|P_A - P_B|",
) -> plt.Figure:
    """Render a 12x12 AST transition heatmap (delta |P_A - P_B| or single file P)."""
    size = len(state_names)
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(transition_matrix, cmap="YlOrRd", aspect="auto", vmin=0.0)

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(state_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(state_names, fontsize=8)
    ax.set_xlabel("Next AST Structural State")
    ax.set_ylabel("Current AST Structural State")
    ax.set_title(title)

    max_val = max(
        (val for row in transition_matrix for val in row),
        default=0.0,
    )
    threshold = max_val * 0.55 if max_val > 0 else 0.5

    for row_idx, row in enumerate(transition_matrix):
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
    cbar.set_label(colorbar_label)
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
  <style>
    body {{
      margin: 0;
      padding: 0;
      background: transparent;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }}
    .diff-loading {{
      display: flex;
      align-items: center;
      gap: 0.65rem;
      padding: 0.75rem 0.25rem;
      color: #31333f;
      font-size: 0.95rem;
    }}
    .st-spinner-ring {{
      width: 1.15rem;
      height: 1.15rem;
      border: 2.5px solid rgba(255, 75, 75, 0.22);
      border-top-color: #ff4b4b;
      border-radius: 50%;
      animation: st-spin 0.75s linear infinite;
      flex-shrink: 0;
    }}
    @keyframes st-spin {{
      to {{
        transform: rotate(360deg);
      }}
    }}
    @media (prefers-color-scheme: dark) {{
      .diff-loading {{
        color: #fafafa;
      }}
    }}
    .d2h-wrapper {{
      margin: 0;
    }}
  </style>
</head>
<body>
  <div id="diff-loading" class="diff-loading" role="status" aria-live="polite">
    <span class="st-spinner-ring" aria-hidden="true"></span>
    <span>Loading code diff...</span>
  </div>
  <div id="diff-target" style="display:none;"></div>
  <script>
    const diffString = {diff_js_literal};
    const loadingElement = document.getElementById("diff-loading");
    const targetElement = document.getElementById("diff-target");

    function loadStylesheet(href) {{
      return new Promise((resolve) => {{
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = href;
        link.onload = resolve;
        link.onerror = resolve;
        document.head.appendChild(link);
      }});
    }}

    function loadScript(src) {{
      return new Promise((resolve, reject) => {{
        const script = document.createElement("script");
        script.src = src;
        script.onload = resolve;
        script.onerror = () => reject(new Error("Failed to load diff viewer script."));
        document.head.appendChild(script);
      }});
    }}

    Promise.allSettled([
      loadStylesheet({json.dumps(HLJS_CSS_URL)}),
      loadStylesheet({json.dumps(DIFF2HTML_CSS_URL)}),
      loadScript({json.dumps(DIFF2HTML_JS_URL)})
    ]).then(() => {{
      if (typeof Diff2HtmlUI === "undefined") {{
        loadingElement.textContent = "Could not load diff viewer assets.";
        return;
      }}
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
      loadingElement.style.display = "none";
      targetElement.style.display = "block";
    }});
  </script>
</body>
</html>
"""


def _init_session_state() -> None:
    """Initialize session state keys on first page load."""
    if "detection_mode_selector" not in st.session_state:
        st.session_state["detection_mode_selector"] = MODE_PAIRWISE_PLAGIARISM
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
    if "last_pair_key" not in st.session_state:
        st.session_state["last_pair_key"] = None
    if "cached_pair_views" not in st.session_state:
        st.session_state["cached_pair_views"] = None
    if "last_predict_key" not in st.session_state:
        st.session_state["last_predict_key"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None
    if "code_inputs_expanded" not in st.session_state:
        st.session_state["code_inputs_expanded"] = False
    if "code_inputs_epoch" not in st.session_state:
        st.session_state["code_inputs_epoch"] = 0


def _on_source_change() -> None:
    """Sync Code A and Code B when the user changes the Source selector."""
    selected = st.session_state["source_selector"]
    st.session_state["code_inputs_expanded"] = selected == MANUAL_SOURCE_LABEL
    st.session_state["code_inputs_epoch"] = int(st.session_state.get("code_inputs_epoch", 0)) + 1
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
    pending = st.session_state.pop("_pending_health", None)
    if pending is not None:
        health, health_error = pending
    else:
        health, health_error = fetch_backend_health(API_URL)
    color, tooltip, is_green = _classify_health(health, health_error)
    _render_api_status_row(color, tooltip)
    if is_green:
        st.rerun()


def _extract_model_inventory(
    task_health: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    """Parse available models and display names (without version numbers) from task health."""
    available_raw = task_health.get("available_models")
    available_models = [
        item for item in (available_raw or []) if isinstance(item, dict) and item.get("id")
    ]
    default_model = (
        str(task_health["default_model"])
        if task_health.get("default_model")
        else None
    )
    model_by_id = {str(item["id"]): item for item in available_models}
    model_labels = {
        str(item["id"]): str(item.get("display_name") or item["id"])
        for item in available_models
    }
    model_ids = list(model_by_id.keys())
    if default_model in model_ids:
        model_ids = [default_model, *[mid for mid in model_ids if mid != default_model]]
    return model_by_id, model_labels, model_ids


def _render_sidebar() -> tuple[str, str, str | None, bool, dict[str, Any]]:
    """Render sidebar API status, Detection Mode, and mode scoped controls."""
    health, health_error = fetch_backend_health(API_URL)
    color, tooltip, is_green = _classify_health(health, health_error)

    tasks = health.get("tasks") if isinstance(health, dict) and isinstance(health.get("tasks"), dict) else {}
    pairwise_health = (
        tasks.get("pairwise_plagiarism")
        if isinstance(tasks.get("pairwise_plagiarism"), dict)
        else (health if isinstance(health, dict) else {})
    )
    ai_health = (
        tasks.get("ai_plagiarism")
        if isinstance(tasks.get("ai_plagiarism"), dict)
        else {}
    )

    with st.sidebar:
        if is_green:
            _render_api_status_row(color, tooltip)
        else:
            st.session_state["_pending_health"] = (health, health_error)
            _poll_api_status_until_ready()

        st.divider()
        st.header("Controls")

        detection_mode = st.selectbox(
            "Detection Mode",
            options=[MODE_PAIRWISE_PLAGIARISM, MODE_AI_PLAGIARISM],
            key="detection_mode_selector",
            filter_mode=None,
        )

        selected_model_id: str | None = None
        language = "java"

        if detection_mode == MODE_PAIRWISE_PLAGIARISM:
            model_by_id, model_labels, model_ids = _extract_model_inventory(pairwise_health)
            if model_ids:
                current_model = st.session_state.get("pairwise_model_selector")
                if current_model not in model_ids:
                    st.session_state["pairwise_model_selector"] = model_ids[0]
                selected_model_id = st.selectbox(
                    "Model",
                    options=model_ids,
                    key="pairwise_model_selector",
                    format_func=lambda mid: model_labels.get(mid, mid),
                    help="Select the pairwise plagiarism classifier used for scoring.",
                    filter_mode=None,
                )
            else:
                st.selectbox(
                    "Model",
                    options=["Default (waiting for API)"],
                    index=0,
                    disabled=True,
                    help="Available models populate once the backend is ready.",
                    filter_mode=None,
                )

            selected_meta = model_by_id.get(selected_model_id) if selected_model_id else None
            supported_languages = (
                selected_meta.get("supported_languages")
                if isinstance(selected_meta, dict) and selected_meta.get("supported_languages")
                else (health.get("supported_languages") if isinstance(health, dict) else None)
            ) or ["java"]

            language_options = list(dict.fromkeys(supported_languages)) or ["java"]
            language = st.selectbox(
                "Language",
                options=language_options,
                index=0,
                key="pairwise_language_selector",
                format_func=lambda lang: lang.upper() if lang == "cpp" else lang.title(),
                help="Calibrated languages for the selected model.",
                filter_mode=None,
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
                filter_mode=None,
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
        else:
            ai_model_by_id, ai_model_labels, ai_model_ids = _extract_model_inventory(ai_health)
            if ai_health.get("model_loaded") and ai_model_ids:
                current_ai_model = st.session_state.get("ai_model_selector")
                if current_ai_model not in ai_model_ids:
                    st.session_state["ai_model_selector"] = ai_model_ids[0]
                selected_model_id = st.selectbox(
                    "Model",
                    options=ai_model_ids,
                    key="ai_model_selector",
                    format_func=lambda mid: ai_model_labels.get(mid, mid),
                    help="Select the AI plagiarism classifier used for scoring.",
                    filter_mode=None,
                )
                ai_meta = ai_model_by_id.get(selected_model_id) if selected_model_id else None
                ai_languages = (
                    ai_meta.get("supported_languages")
                    if isinstance(ai_meta, dict) and ai_meta.get("supported_languages")
                    else None
                ) or ["java"]
                language = st.selectbox(
                    "Language",
                    options=list(dict.fromkeys(ai_languages)) or ["java"],
                    index=0,
                    key="ai_language_selector",
                    format_func=lambda lang: lang.upper() if lang == "cpp" else lang.title(),
                    help="Calibrated languages for the selected model.",
                    filter_mode=None,
                )
            else:
                st.selectbox(
                    "Model",
                    options=["No model loaded"],
                    index=0,
                    disabled=True,
                    key="ai_model_selector_disabled",
                    filter_mode=None,
                )
                st.selectbox(
                    "Language",
                    options=["java"],
                    index=0,
                    disabled=True,
                    key="ai_language_selector_disabled",
                    format_func=lambda lang: lang.upper() if lang == "cpp" else lang.title(),
                    filter_mode=None,
                )

    return detection_mode, language, selected_model_id, is_green, ai_health


def _render_verdict_kpi(
    prob: float,
    predicted_label: int,
    positive_label: str = "Plagiarized",
    negative_label: str = "Clean",
    prob_label: str = "Plagiarism Probability",
) -> None:
    """Render Verdict and Probability metrics with shared color and spacing styles."""
    is_positive = predicted_label == 1
    verdict_text = positive_label if is_positive else negative_label
    container_key = "primary_kpi_plag" if is_positive else "primary_kpi_clean"

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
          .st-key-baseline_kpi_grid [data-testid="stColumn"] [data-testid="stVerticalBlock"] {
            gap: 0.15rem;
          }
          .st-key-baseline_kpi_grid [data-testid="stMetric"] {
            padding-bottom: 0;
          }
          .st-key-baseline_kpi_grid [data-testid="stCaptionContainer"] {
            margin-top: -0.25rem;
            margin-bottom: 0.45rem;
          }
          .st-key-baseline_kpi_grid [data-testid="stCaptionContainer"] p {
            margin-top: 0;
            margin-bottom: 0;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key=container_key):
        kpi_col_a, kpi_col_b = st.columns(2, gap="medium")
        kpi_col_a.metric("Verdict", verdict_text)
        kpi_col_b.metric(prob_label, f"{prob * 100:.1f}%")


def _format_metric_value(raw_value: float, fmt: str) -> str:
    """Format a baseline metric value for display."""
    if fmt == "pct":
        return f"{raw_value:.1f}%"
    if fmt == "int":
        return f"{int(round(raw_value))}"
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


def _render_kpi_panel(
    kpi_slot: Any,
    result: dict[str, Any],
) -> None:
    """Render Verdict, Plagiarism Probability, and baseline KPI metrics into a slot."""
    prob = float(result.get("plagiarism_probability", 0.0))
    predicted_label = int(result.get("predicted_label", 0))

    with kpi_slot.container():
        _render_verdict_kpi(
            prob=prob,
            predicted_label=predicted_label,
        )

        with st.container(key="baseline_kpi_grid"):
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


def _render_ai_kpi_panel(
    kpi_slot: Any,
    result: dict[str, Any],
) -> None:
    """Render AI Plagiarism verdict, probability, and single file AST KPIs into a slot."""
    ai_prob = float(result.get("ai_probability", 0.0))
    predicted_label = int(result.get("predicted_label", 0))

    with kpi_slot.container():
        _render_verdict_kpi(
            prob=ai_prob,
            predicted_label=predicted_label,
            positive_label="AI Plagiarized",
            negative_label="Human Written",
            prob_label="AI Plagiarism Probability",
        )

        with st.container(key="baseline_kpi_grid"):
            row_cols = st.columns(2, gap="medium")
            for col, spec in zip(row_cols, AI_BASELINE_METRIC_SPECS):
                val = float(result.get(spec["key"], 0.0))
                with col:
                    st.metric(
                        label=spec["label"],
                        value=_format_metric_value(val, spec["format"]),
                        help=spec["tooltip"],
                    )
                    st.caption(spec["description"])


def _render_heatmap_png(
    result: dict[str, Any],
    matrix_key: str = "transition_delta",
    title: str = "AST Transition Delta Heatmap (|P_A - P_B|)",
    colorbar_label: str = "|P_A - P_B|",
) -> bytes | None:
    """Validate transition matrix dimensions and rasterize the heatmap to PNG bytes."""
    state_names = list(result.get("state_names", []))
    matrix = list(result.get(matrix_key, []))
    if not (
        state_names
        and matrix
        and len(matrix) == len(state_names)
        and all(len(row) == len(state_names) for row in matrix)
    ):
        return None

    fig = render_transition_heatmap(
        state_names,
        matrix,
        title=title,
        colorbar_label=colorbar_label,
    )
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
        return buf.getvalue()
    finally:
        plt.close(fig)


def _paint_heatmap_panel(heatmap_slot: Any, heatmap_png: bytes | None) -> None:
    """Render pre-rendered heatmap PNG bytes or a mismatch warning into a slot."""
    with heatmap_slot.container():
        if heatmap_png is not None:
            st.image(heatmap_png, width="stretch")
        else:
            st.warning(
                "Transition matrix data was empty or mismatched. "
                "The heatmap did not render."
            )


def _prepare_diff_payload(
    code_a: str,
    code_b: str,
    language: str,
) -> tuple[str, int] | None:
    """Build diff2html HTML and frame height, or return None when inputs match."""
    unified_text = build_unified_diff(code_a, code_b, language=language)
    if not unified_text:
        return None
    diff_html = build_diff2html_document(unified_text)
    line_count = max(len(code_a.splitlines()), len(code_b.splitlines()), 10)
    frame_height = min(max(line_count * 24 + 90, 320), 760)
    return diff_html, frame_height


def _paint_diff_panel(
    diff_slot: Any,
    diff_payload: tuple[str, int] | None,
) -> None:
    """Render the side by side diff iframe or an identical-code notice into a slot."""
    with diff_slot.container():
        if diff_payload is not None:
            diff_html, frame_height = diff_payload
            components.html(diff_html, height=frame_height, scrolling=True)
        else:
            st.info("Code A and Code B are identical.")


def _render_empty_placeholders(
    kpi_slot: Any,
    heatmap_slot: Any,
    diff_slot: Any,
    has_a: bool,
    has_b: bool,
) -> None:
    """Render bare guidance captions when one or both code inputs are empty."""
    if not has_a and not has_b:
        kpi_slot.caption("Select a source or add Code A and Code B to view metrics.")
        heatmap_slot.caption("Select a source or add Code A and Code B to view the heatmap.")
        diff_slot.caption("Select a source or add Code A and Code B to view the diff.")
        return

    missing_label = "Code A" if not has_a else "Code B"
    kpi_slot.caption(f"Add {missing_label} to view metrics.")
    heatmap_slot.caption(f"Add {missing_label} to view the heatmap.")
    diff_slot.caption(f"Add {missing_label} to view the diff.")


def _render_unavailable_placeholders(
    kpi_slot: Any,
    heatmap_slot: Any,
    diff_slot: Any,
) -> None:
    """Render bare fallback captions when analysis fails with an API or input error."""
    kpi_slot.caption("Metrics unavailable.")
    heatmap_slot.caption("Heatmap unavailable.")
    diff_slot.caption("Diff unavailable.")


def _fetch_and_render_kpi(
    kpi_slot: Any,
    code_a: str,
    code_b: str,
    language: str,
    model_id: str | None,
    predict_key: Any,
) -> dict[str, Any] | None:
    """Call POST /predict/pairwise_plagiarism, update cache state, and render KPI panel."""
    result, status_code, error_detail = request_prediction(
        api_url=API_URL,
        code_a=code_a,
        code_b=code_b,
        language=language,
        model=model_id,
    )
    if result is not None:
        st.session_state["last_predict_key"] = predict_key
        st.session_state["last_result"] = {
            "result": result,
            "code_a": code_a,
            "code_b": code_b,
            "language": language,
        }
        st.session_state["last_error"] = None
        _render_kpi_panel(kpi_slot, result)
        return result

    st.session_state["last_predict_key"] = None
    st.session_state["last_result"] = None
    st.session_state["last_error"] = (status_code, error_detail)
    return None


def _run_analysis_into_slots(
    code_a: str,
    code_b: str,
    language: str,
    model_id: str | None,
    backend_ready: bool,
    kpi_slot: Any,
    heatmap_slot: Any,
    diff_slot: Any,
) -> None:
    """Run prediction if inputs changed and populate section slots progressively."""
    if not code_a.strip() or not code_b.strip():
        st.session_state["last_result"] = None
        st.session_state["last_pair_key"] = None
        st.session_state["cached_pair_views"] = None
        st.session_state["last_predict_key"] = None
        st.session_state["last_error"] = None
        _render_empty_placeholders(
            kpi_slot=kpi_slot,
            heatmap_slot=heatmap_slot,
            diff_slot=diff_slot,
            has_a=bool(code_a.strip()),
            has_b=bool(code_b.strip()),
        )
        return

    pair_key = (_content_hash(code_a), _content_hash(code_b), language)
    predict_key = (pair_key, model_id)
    pair_cached = (
        pair_key == st.session_state.get("last_pair_key")
        and st.session_state.get("cached_pair_views") is not None
    )

    if predict_key == st.session_state.get("last_predict_key"):
        saved = st.session_state.get("last_result")
        if saved is not None and pair_cached:
            views = st.session_state["cached_pair_views"]
            _render_kpi_panel(kpi_slot, saved["result"])
            _paint_heatmap_panel(heatmap_slot, views["heatmap_png"])
            _paint_diff_panel(diff_slot, views["diff_payload"])
            return

    if not backend_ready:
        st.session_state["last_predict_key"] = None
        st.session_state["last_result"] = None
        st.session_state["last_error"] = (503, "Model service is warming up or unavailable.")
        _render_unavailable_placeholders(kpi_slot, heatmap_slot, diff_slot)
        return

    if pair_cached:
        views = st.session_state["cached_pair_views"]
        _paint_heatmap_panel(heatmap_slot, views["heatmap_png"])
        _paint_diff_panel(diff_slot, views["diff_payload"])
        with kpi_slot.spinner("Computing analysis metrics..."):
            result = _fetch_and_render_kpi(
                kpi_slot=kpi_slot,
                code_a=code_a,
                code_b=code_b,
                language=language,
                model_id=model_id,
                predict_key=predict_key,
            )
            if result is None:
                kpi_slot.caption("Metrics unavailable.")
        return

    result: dict[str, Any] | None = None
    with diff_slot.spinner("Preparing code diff..."):
        with heatmap_slot.spinner("Rendering transition heatmap..."):
            with kpi_slot.spinner("Computing analysis metrics..."):
                result = _fetch_and_render_kpi(
                    kpi_slot=kpi_slot,
                    code_a=code_a,
                    code_b=code_b,
                    language=language,
                    model_id=model_id,
                    predict_key=predict_key,
                )

            heatmap_png: bytes | None = None
            if result is not None:
                heatmap_png = _render_heatmap_png(result)
                _paint_heatmap_panel(heatmap_slot, heatmap_png)

        if result is not None:
            diff_payload = _prepare_diff_payload(code_a, code_b, language)
            _paint_diff_panel(diff_slot, diff_payload)
            st.session_state["last_pair_key"] = pair_key
            st.session_state["cached_pair_views"] = {
                "heatmap_png": heatmap_png,
                "diff_payload": diff_payload,
            }

    if result is None:
        _render_unavailable_placeholders(kpi_slot, heatmap_slot, diff_slot)


def _render_pairwise_plagiarism_view(
    language: str,
    model_id: str | None,
    backend_ready: bool,
) -> None:
    """Render the Pairwise Plagiarism main view below the page title."""
    epoch = int(st.session_state.get("code_inputs_epoch", 0))
    code_inputs_expander = st.expander(
        "Code Inputs",
        expanded=bool(st.session_state.get("code_inputs_expanded", False)),
        key=f"code_inputs_expander_{epoch}",
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

    parse_warning_slot = st.empty()
    error_slot = st.empty()

    st.divider()
    st.subheader("Analysis Results")
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        kpi_slot = st.empty()
    with right_col:
        heatmap_slot = st.empty()

    st.divider()
    st.subheader("Code Diff")
    diff_slot = st.empty()

    _run_analysis_into_slots(
        code_a=code_a,
        code_b=code_b,
        language=language,
        model_id=model_id,
        backend_ready=backend_ready,
        kpi_slot=kpi_slot,
        heatmap_slot=heatmap_slot,
        diff_slot=diff_slot,
    )

    has_a = bool(code_a.strip())
    has_b = bool(code_b.strip())
    saved = st.session_state.get("last_result")
    last_error = st.session_state.get("last_error")

    if has_b and not has_a:
        with hint_slot_a:
            _render_inline_hint("Add code to Code A to run the comparison.", tone="amber")
    if has_a and not has_b:
        with hint_slot_b:
            _render_inline_hint("Add code to Code B to run the comparison.", tone="amber")

    parse_warnings: list[str] = []
    if saved is not None:
        if bool(saved["result"].get("parse_warnings_a", False)):
            parse_warnings.append("Code A: Syntax error found. Partial tree used.")
        if bool(saved["result"].get("parse_warnings_b", False)):
            parse_warnings.append("Code B: Syntax error found. Partial tree used.")
    if parse_warnings:
        with parse_warning_slot.container():
            for msg in parse_warnings:
                _render_inline_hint(msg, tone="amber")

    if last_error is not None:
        status_code, error_detail = last_error
        with error_slot.container():
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


def _render_ai_plagiarism_view(
    ai_health: dict[str, Any],
    language: str,
    model_id: str | None,
) -> None:
    """Render the AI Plagiarism main view matching Pairwise layout without Code B or Code Diff."""
    ai_model_loaded = bool(ai_health.get("model_loaded"))

    code_inputs_expander = st.expander(
        "Code Inputs",
        expanded=False,
        key="ai_code_inputs_expander",
    )
    with code_inputs_expander:
        submission_code = st.text_area(
            "Code",
            key="ai_plagiarism_code",
            height=340,
            disabled=not ai_model_loaded,
        )

    parse_warning_slot = st.empty()
    error_slot = st.empty()

    st.divider()
    st.subheader("Analysis Results")
    left_col, right_col = st.columns(2, gap="large")
    with left_col:
        kpi_slot = st.empty()
    with right_col:
        heatmap_slot = st.empty()

    if not ai_model_loaded:
        kpi_slot.caption("No AI plagiarism model is loaded yet.")
        heatmap_slot.caption("Heatmap unavailable.")
        return

    if not submission_code.strip():
        kpi_slot.caption("Add Code to view metrics.")
        heatmap_slot.caption("Add Code to view the heatmap.")
        return

    result: dict[str, Any] | None = None
    status_code: int | None = None
    error_detail: str | None = None
    with heatmap_slot.spinner("Rendering transition heatmap..."):
        with kpi_slot.spinner("Evaluating AI plagiarism..."):
            result, status_code, error_detail = request_ai_plagiarism_prediction(
                api_url=API_URL,
                code=submission_code,
                language=language,
                model=model_id,
            )
            if result is not None:
                _render_ai_kpi_panel(kpi_slot, result)

        if result is not None:
            heatmap_png = _render_heatmap_png(
                result,
                matrix_key="transition_matrix",
                title="AST Transition Matrix Heatmap (P)",
                colorbar_label="P(next | current)",
            )
            _paint_heatmap_panel(heatmap_slot, heatmap_png)

    if result is None:
        kpi_slot.caption("Metrics unavailable.")
        heatmap_slot.caption("Heatmap unavailable.")
        with error_slot.container():
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
        return

    if bool(result.get("parse_warnings", False)):
        with parse_warning_slot.container():
            _render_inline_hint("Code: Syntax error found. Partial tree used.", tone="amber")


def main() -> None:
    """Entrypoint for the Streamlit application."""
    st.set_page_config(
        page_title="Code Plagiarism Detector",
        layout="wide",
    )
    _init_session_state()

    (
        detection_mode,
        language,
        model_id,
        backend_ready,
        ai_health,
    ) = _render_sidebar()

    st.title("Code Plagiarism Detector")

    if detection_mode == MODE_PAIRWISE_PLAGIARISM:
        _render_pairwise_plagiarism_view(
            language=language,
            model_id=model_id,
            backend_ready=backend_ready,
        )
    else:
        _render_ai_plagiarism_view(
            ai_health=ai_health,
            language=language,
            model_id=model_id,
        )


if __name__ == "__main__":
    main()
