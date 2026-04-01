#!/usr/bin/env bash
# The console's gate, as a file rather than an inline pre-commit `entry`:
# the command contains colons and quotes, which YAML mangles.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d node_modules ]; then
  echo "console/node_modules missing — skipping. Run: just console-install"
  exit 0
fi
npm run --silent ci
