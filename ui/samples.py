"""Static Java demo functions for the Streamlit dashboard.

All four samples implement the same function (getRemainingPath): one version written by a human
and one by each of three LLMs. They come from the test split of the HumanVsAICode dataset
(https://huggingface.co/datasets/OSS-forge/HumanVsAICode, license CC-BY-SA-4.0, Cotroneo, Improta and
Liguori, ISSRE 2025), so the model never trained on them. They are embedded as static strings so the
UI never reads raw dataset files at runtime. The detector is not perfect, so it may misjudge some of them.
"""

from __future__ import annotations

DEMO_SAMPLES: dict[str, dict[str, str]] = {
    "Human (GitHub)": {
        "title": "Human (GitHub)",
        "description": "Same function, written by a human developer.",
        "expected_label": "Human",
        "language": "java",
        "code": """protected static String getRemainingPath(List<String> tokens, int i, int end) {
        if (tokens == null) {
            throw MESSAGES.nullArgument("tokens");
        }
        if (i < 0 || i >= end) { throw new IllegalArgumentException("i is not in the range of tokens: 0-" + (end - 1)); }
        if (i == end - 1) { return tokens.get(end - 1); }
        StringBuilder buffer = new StringBuilder();
        for (; i < end - 1; ++i) {
            buffer.append(tokens.get(i));
            buffer.append("/");
        }
        buffer.append(tokens.get(end - 1));
        return buffer.toString();
    }""",
    },
    "ChatGPT (gpt-3.5-turbo)": {
        "title": "ChatGPT (gpt-3.5-turbo)",
        "description": "Same function, written by an AI model.",
        "expected_label": "AI",
        "language": "java",
        "code": """public static String getRemainingPath(String[] tokens, int i, int end) throws IllegalArgumentException{
    if(tokens == null || i < 0 || i >= tokens.length || end < i || end >= tokens.length){
        throw new IllegalArgumentException();
    }
    StringBuilder sb = new StringBuilder();
    for(int j=i; j<=end; j++){
        sb.append(tokens[j]);
        if(j < end){
            sb.append("/");
        }
    }
    return sb.toString();
}""",
    },
    "DeepSeek-Coder 33B": {
        "title": "DeepSeek-Coder 33B",
        "description": "Same function, written by an AI model.",
        "expected_label": "AI",
        "language": "java",
        "code": """protected static String getRemainingPath(List<String> tokens, int i, int end) {
    if (tokens == null || i < 0 || i >= tokens.size() || end < 0 || end >= tokens.size() || i > end) {
        throw new IllegalArgumentException("Invalid arguments");
    }

    StringBuilder sb = new StringBuilder();
    for (int j = i; j <= end; j++) {
        sb.append("/");
        sb.append(tokens.get(j));
    }
    return sb.toString();
}""",
    },
    "Qwen2.5-Coder 32B": {
        "title": "Qwen2.5-Coder 32B",
        "description": "Same function, written by an AI model.",
        "expected_label": "AI",
        "language": "java",
        "code": """protected static String getRemainingPath(List<String> tokens, int i, int end) {
	if (tokens == null) {
        throw new IllegalArgumentException("tokens cannot be null");
    }
    if (i < 0 || i > end || end > tokens.size()) {
        throw new IllegalArgumentException("i or end is out of range");
    }
    StringBuilder path = new StringBuilder();
    for (int j = i; j <= end; j++) {
        path.append(tokens.get(j));
        if (j < end) {
            path.append("/");
        }
    }
    return path.toString();
}""",
    },
}
