#!/usr/bin/env bash
# The console's gate, as a file rather than an inline pre-commit `entry`:
# the command contains colons and quotes, which YAML mangles.
set -euo pipefail
cd "$(dirname "$0")"
# Refuse rather than skip. A Python-only checkout is already spared by the
# hook's `files: ^console/` scoping, so exiting 0 here never protected that
# case -- it only fired for someone editing the console with no deps
# installed, which is precisely when the gate is worth running. A check that
# passes without running is the same failure as a test that asserts nothing.
if [ ! -d node_modules ]; then
  echo "console/check.sh: node_modules missing, so the console was NOT checked." >&2
  echo "Run: just console-install" >&2
  exit 1
fi
npm run --silent ci
