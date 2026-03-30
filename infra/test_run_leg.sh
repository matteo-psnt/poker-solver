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

echo "=== case 6: an abstraction ALREADY on the share is never republished ==="
# Regression: keying the skip on the completion marker alone made every
# precompute leg rm -rf and re-upload marker-less abstractions that push-data
# had put there -- live artifacts, ~773 MB, on every run.
rm -rf "$WORK/data/combo_abstraction" "$SHARE/combo_abstraction"
mkdir -p "$WORK/data/combo_abstraction" "$SHARE/combo_abstraction"
# A legacy, marker-LESS abstraction present on both share and node.
mkdir -p "$SHARE/combo_abstraction/buckets-LEGACY-old111" "$WORK/data/combo_abstraction/buckets-LEGACY-old111"
echo original > "$SHARE/combo_abstraction/buckets-LEGACY-old111/payload"
echo original > "$WORK/data/combo_abstraction/buckets-LEGACY-old111/payload"
bash "$LEG" >"$ROOT/out6.log" 2>&1
check "legacy dir survives untouched" "$(cat "$SHARE/combo_abstraction/buckets-LEGACY-old111/payload")" "original"
check "no marker invented for it" "$([ -f "$SHARE/combo_abstraction/.complete-buckets-LEGACY-old111" ] && echo yes || echo no)" "no"
check "only the NEW abstraction published" "$(grep -c 'precompute complete: 1' "$ROOT/out6.log")" "1"

echo "=== case 7: ab mode refuses to guess its preconditions ==="
# The harness's whole value is that seed and arms cannot be omitted. If the leg
# silently defaulted either, a cloud A/B would look identical to a local one and
# be quietly incomparable.
rm -rf "$WORK/data/combo_abstraction"; mkdir -p "$WORK/data/combo_abstraction"
ARMS_HEX=$(printf '%s' "a:solver__cfr_plus=true" | python3 -c "import sys; print(sys.stdin.read().encode().hex())")

RUN_OP=ab RUN_TO=100 RUN_AB_ARMS_HEX="$ARMS_HEX" bash "$LEG" >"$ROOT/out7a.log" 2>&1; rc=$?
check "no seed -> nonzero exit" "$([ "$rc" != 0 ] && echo yes || echo no)" "yes"
check "no seed -> says so" "$(grep -c 'needs RUN_AB_SEED' "$ROOT/out7a.log")" "1"

RUN_OP=ab RUN_TO=100 RUN_AB_SEED=42 bash "$LEG" >"$ROOT/out7b.log" 2>&1; rc=$?
check "no arms -> nonzero exit" "$([ "$rc" != 0 ] && echo yes || echo no)" "yes"
check "no arms -> says so" "$(grep -c 'needs RUN_AB_ARMS_HEX' "$ROOT/out7b.log")" "1"

# And with both present it reaches the CLI with the arms decoded.
cat > "$BIN/uv" <<'STUB2'
#!/usr/bin/env bash
for a in "$@"; do [ "$a" = "ab" ] && { echo "AB-INVOKED: $*"; exit 0; }; done
exit 0
STUB2
chmod +x "$BIN/uv"
RUN_OP=ab RUN_TO=100 RUN_AB_SEED=42 RUN_AB_ARMS_HEX="$ARMS_HEX" bash "$LEG" >"$ROOT/out7c.log" 2>&1
check "arms decoded and passed through" "$(grep -c -- '--arm a:solver__cfr_plus=true' "$ROOT/out7c.log")" "1"
check "seed passed through" "$(grep -c -- '--seed 42' "$ROOT/out7c.log")" "1"

echo "=== case 5: missing RUN_CONFIG fails loudly ==="
RUN_CONFIG="" bash "$LEG" >"$ROOT/out5.log" 2>&1; rc=$?
check "nonzero exit" "$([ "$rc" != 0 ] && echo yes || echo no)" "yes"
check "explains why" "$(grep -c 'needs RUN_CONFIG' "$ROOT/out5.log")" "1"

echo
[ "$fail" = 0 ] && echo "ALL PASS" || { echo "FAILURES"; tail -20 "$ROOT/out1.log"; }
exit "$fail"
