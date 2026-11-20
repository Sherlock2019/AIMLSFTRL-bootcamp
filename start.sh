#!/usr/bin/env bash
# Start the AI Training Lab dashboard on http://localhost:${PORT:-8810}
#   ./start.sh            start (creates .venv on first run)
#   ./start.sh --stop     stop the dashboard
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8810}"

if [ "${1:-}" = "--stop" ]; then
  pid=$(ss -ltnp 2>/dev/null | grep ":$PORT " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
  [ -n "$pid" ] && kill "$pid" && echo "Stopped dashboard (pid $pid)" || echo "Nothing listening on $PORT"
  exit 0
fi

[ -x .venv/bin/python ] || bash scripts/setup_env.sh
if ! .venv/bin/python -c "from lab import hardware; import sys; sys.exit(0 if hardware.model_cached('${LAB_BASE_MODEL:-Qwen/Qwen3-0.6B}') else 1)"; then
  echo "Downloading base model (one-time)…"
  .venv/bin/python scripts/download_model.py "${LAB_BASE_MODEL:-Qwen/Qwen3-0.6B}"
fi
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "AI Training Lab → http://localhost:$PORT"
exec .venv/bin/streamlit run app.py --server.port "$PORT"
