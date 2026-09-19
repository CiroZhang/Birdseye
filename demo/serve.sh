#!/bin/bash
# Serves this folder locally and opens it in your default browser.
cd "$(dirname "$0")"
PORT=8080
echo "Serving at http://localhost:$PORT"
open "http://localhost:$PORT" 2>/dev/null
python3 -m http.server "$PORT"
