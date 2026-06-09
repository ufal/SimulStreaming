#!/usr/bin/env bash
set -euo pipefail

REGISTRY_URL="${REGISTRY_URL:-http://localhost:8003}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"

DISPLAY_NAME="${1:?display name required}"
STREAM_PATH="${2:?stream path required}"
shift 2

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "No ffmpeg args provided" >&2
  exit 2
fi

CLEANED_UP=0
FFMPEG_PID=""

cleanup() {
  if [[ "$CLEANED_UP" -eq 1 ]]; then
    return
  fi
  CLEANED_UP=1

  if [[ -n "${FFMPEG_PID:-}" ]] && kill -0 "$FFMPEG_PID" 2>/dev/null; then
    kill "$FFMPEG_PID" 2>/dev/null || true
    wait "$FFMPEG_PID" 2>/dev/null || true
  fi

  curl -fsS -X POST "$REGISTRY_URL/stream/unregister" \
    -H "Content-Type: application/json" \
    -d "$(python3 - <<'PY' "$STREAM_PATH"
import json, sys
print(json.dumps({"path": sys.argv[1]}))
PY
)" >/dev/null 2>&1 || true
}

on_signal() {
  cleanup
  exit 130
}

trap on_signal INT TERM QUIT
trap cleanup EXIT

curl -fsS -X POST "$REGISTRY_URL/stream/register" \
  -H "Content-Type: application/json" \
  -d "$(python3 - <<'PY' "$STREAM_PATH" "$DISPLAY_NAME"
import json, sys
payload = {
    "path": sys.argv[1],
    "display_name": sys.argv[2],
    "source_type": "ffmpeg",
    "source_id": sys.argv[1],
}
print(json.dumps(payload, ensure_ascii=False))
PY
)" >/dev/null

"$FFMPEG_BIN" "$@" &
FFMPEG_PID=$!

wait "$FFMPEG_PID"
exit_code=$?
exit "$exit_code"
