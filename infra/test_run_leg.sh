#!/usr/bin/env bash
# Exercise the REAL run_leg.sh precompute branch off-node.
#
#   bash infra/test_run_leg.sh infra/run_leg.sh
#
# Runs in seconds against temp dirs; no Azure, no node, no compute.
#
# Stubs the two things that need a node: `uv` (so no dependency sync or actual
# precompute runs) and the code staging. Everything else -- the publish ordering,
# the completion marker, the marker-gated refresh -- is the shipped code.
set -uo pipefail

LEG="$1"
ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT

export WORK="$ROOT/work"
export AZ_BATCH_NODE_MOUNTS_DIR="$ROOT/mounts"
SHARE="$AZ_BATCH_NODE_MOUNTS_DIR/shared"
mkdir -p "$WORK/data/combo_abstraction" "$SHARE/combo_abstraction" "$WORK/code"

# --- stub `uv`: pretend precompute produced an abstraction on local disk ------
BIN="$ROOT/bin"; mkdir -p "$BIN"
cat > "$BIN/uv" <<STUB
#!/usr/bin/env bash
# "uv sync" -> no-op; "uv run poker-solver-run precompute --config X" -> emit dir
for a in "\$@"; do
  if [ "\$a" = "precompute" ]; then
    d="$WORK/data/combo_abstraction/buckets-TEST-abc123"
    mkdir -p "\$d"
    echo '{"config_hash":"abc123"}' > "\$d/metadata.json"
    dd if=/dev/zero of="\$d/big.npy" bs=1024 count=64 2>/dev/null
    echo "  precomputed -> \$d"
    exit 0
  fi
done
exit 0
STUB
chmod +x "$BIN/uv"
export PATH="$BIN:$PATH"

export CODE_DIR="$WORK/code"
export CODE_SNAPSHOT=test-snap
export RUN_OP=precompute
export RUN_CONFIG=some_abstraction
export RUN_TIMEOUT=30s
export AZ_BATCH_TASK_ID=task-test

fail=0
check() { if [ "$2" = "$3" ]; then echo "  PASS $1"; else echo "  FAIL $1: expected '$3', got '$2'"; fail=1; fi; }

echo "=== case 1: precompute publishes with a completion marker ==="
bash "$LEG" >"$ROOT/out1.log" 2>&1; rc=$?
check "exit code" "$rc" "0"
check "published dir on share" "$([ -d "$SHARE/combo_abstraction/buckets-TEST-abc123" ] && echo yes || echo no)" "yes"
check "completion marker written" "$([ -f "$SHARE/combo_abstraction/.complete-buckets-TEST-abc123" ] && echo yes || echo no)" "yes"
check "payload copied" "$([ -f "$SHARE/combo_abstraction/buckets-TEST-abc123/big.npy" ] && echo yes || echo no)" "yes"
check "no staging left behind" "$(ls -d "$SHARE"/combo_abstraction/.staging-* 2>/dev/null | wc -l | tr -d ' ')" "0"

echo "=== case 2: re-running is a no-op (already complete on the share) ==="
before=$(stat -f %m "$SHARE/combo_abstraction/.complete-buckets-TEST-abc123" 2>/dev/null)
sleep 1
bash "$LEG" >"$ROOT/out2.log" 2>&1; rc=$?
check "exit code" "$rc" "0"
check "reports 0 published" "$(grep -c 'precompute complete: 0' "$ROOT/out2.log")" "1"
after=$(stat -f %m "$SHARE/combo_abstraction/.complete-buckets-TEST-abc123" 2>/dev/null)
check "marker untouched" "$before" "$after"

echo "=== case 3: refresh pulls a MARKED abstraction down to a fresh node ==="
rm -rf "$WORK/data/combo_abstraction"; mkdir -p "$WORK/data/combo_abstraction"
bash "$LEG" >"$ROOT/out3.log" 2>&1
check "marked dir refreshed locally" \
  "$([ -f "$WORK/data/combo_abstraction/buckets-TEST-abc123/big.npy" ] && echo yes || echo no)" "yes"

echo "=== case 4: an UNMARKED share dir is NOT consumed (partial publish) ==="
rm -rf "$WORK/data/combo_abstraction"; mkdir -p "$WORK/data/combo_abstraction"
mkdir -p "$SHARE/combo_abstraction/buckets-PARTIAL-def456"
echo '{}' > "$SHARE/combo_abstraction/buckets-PARTIAL-def456/metadata.json"
bash "$LEG" >"$ROOT/out4.log" 2>&1
check "unmarked dir refused" \
  "$([ -d "$WORK/data/combo_abstraction/buckets-PARTIAL-def456" ] && echo yes || echo no)" "no"

echo "=== case 5: missing RUN_CONFIG fails loudly ==="
RUN_CONFIG="" bash "$LEG" >"$ROOT/out5.log" 2>&1; rc=$?
check "nonzero exit" "$([ "$rc" != 0 ] && echo yes || echo no)" "yes"
check "explains why" "$(grep -c 'needs RUN_CONFIG' "$ROOT/out5.log")" "1"

echo
[ "$fail" = 0 ] && echo "ALL PASS" || { echo "FAILURES"; tail -20 "$ROOT/out1.log"; }
exit "$fail"
