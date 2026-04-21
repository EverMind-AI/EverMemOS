#!/usr/bin/env bash
# Phase 4 baseline sweep for the latency-alignment plan.
#
# Step A (latency baseline): concurrency=1 + retry_policy=strict_no_retry.
#   Small sample (30 questions × 3 repeats) is enough for clean p50/p95.
# Step B (throughput baseline): concurrency=default + retry_policy=realistic.
#   Full 1540 questions, production-aligned numbers.
#
# Systems compared: evermemos, openclaw-native-embed, openclaw-native-noembed.
#
# Usage:
#   bash evaluation/scripts/run_latency_baseline.sh step_a
#   bash evaluation/scripts/run_latency_baseline.sh step_b
#   bash evaluation/scripts/run_latency_baseline.sh both
#
# Env needed:
#   LLM_API_KEY, LLM_BASE_URL, SOPH_EMBED_URL (for embed systems), etc.
#   See .env / docs/installation/.

set -euo pipefail

DATASET="locomo"
SYSTEMS=("evermemos" "openclaw-native-embed" "openclaw-native-noembed")
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

STAMP=$(date -u +%Y%m%dT%H%M%S)
RESULTS_ROOT="${RESULTS_ROOT:-/tmp/latency-baseline-$STAMP}"
mkdir -p "$RESULTS_ROOT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

step_a() {
  log "Step A — latency baseline (concurrency=1, strict_no_retry, 30q × 3 repeats)"
  for sys in "${SYSTEMS[@]}"; do
    for run in 1 2 3; do
      OUT="$RESULTS_ROOT/step_a-${sys}-r${run}"
      log "  → $sys run $run → $OUT"
      uv run python -m evaluation.cli \
        --dataset "$DATASET" \
        --system "$sys" \
        --smoke --smoke-messages 0 --smoke-questions 30 \
        --retry-policy strict_no_retry \
        --deadline-ms 120000 \
        --run-name "latency-baseline-a-r${run}" \
        --output-dir "$OUT" \
        2>&1 | tee "$OUT.log"
    done
  done
  log "Step A done. Results root: $RESULTS_ROOT"
}

step_b() {
  log "Step B — throughput baseline (default concurrency, realistic, full 1540q)"
  for sys in "${SYSTEMS[@]}"; do
    OUT="$RESULTS_ROOT/step_b-${sys}"
    log "  → $sys → $OUT"
    uv run python -m evaluation.cli \
      --dataset "$DATASET" \
      --system "$sys" \
      --retry-policy realistic \
      --deadline-ms 600000 \
      --run-name "latency-baseline-b" \
      --output-dir "$OUT" \
      2>&1 | tee "$OUT.log"
  done
  log "Step B done. Results root: $RESULTS_ROOT"
}

case "${1:-both}" in
  step_a) step_a ;;
  step_b) step_b ;;
  both)   step_a; step_b ;;
  *) echo "usage: $0 {step_a|step_b|both}"; exit 2 ;;
esac
