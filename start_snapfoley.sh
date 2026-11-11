#!/usr/bin/env bash
# Start the SnapFoley backend + web UI (runs in foreground - keep this terminal open)
set -e
cd "$(dirname "$0")"
# Make the port public so the shareable URL works without GitHub login
if [ -n "$CODESPACE_NAME" ]; then
  gh codespace ports visibility 7860:public -c "$CODESPACE_NAME" >/dev/null 2>&1 || \
    echo "note: if the site asks you to log in, set port 7860 -> Public in the Ports panel"
fi
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 7860
