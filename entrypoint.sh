#!/usr/bin/env bash
set -euo pipefail

export API_URL="http://127.0.0.1:8000"
PORT="${PORT:-7860}"

uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

python - <<'PY'
import sys
import time
import urllib.request

deadline = time.monotonic() + 30.0
url = "http://127.0.0.1:8000/health"

while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            if response.status == 200:
                sys.exit(0)
    except Exception:
        time.sleep(0.5)

print("Timed out waiting for FastAPI /health readiness after 30s.", file=sys.stderr)
sys.exit(1)
PY

if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "FastAPI process exited unexpectedly." >&2
    exit 1
fi

exec streamlit run ui/app.py \
    --server.address=0.0.0.0 \
    --server.port="${PORT}" \
    --server.headless=true
