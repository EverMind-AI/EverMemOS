#!/usr/bin/env bash
# Claude Code Routine setup script — auto-bench routine.
#
# Runs ONCE per routine session on the cloud container (output cached ~7 days).
# Keep cheap and idempotent.
set -euo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

echo "::group::Python dependencies (evaluation-full)"
uv sync --group evaluation-full
echo "::endgroup::"

echo "::group::Preload EverMemOS docker images"
# EverMemOS infra is NOT started by the routine (candidates don't need it),
# but pulling keeps the cached image layer available for future cold runs.
docker compose -f docker-compose.yaml pull --quiet || \
  echo "  (pull failed — skipping; routine doesn't require EverOS infra)"
echo "::endgroup::"

echo "::group::Candidate scratch dir"
mkdir -p /tmp/candidate
echo "::endgroup::"

echo "::group::Environment sanity"
missing=()
[[ -z "${LLM_API_KEY:-}" ]] && missing+=("LLM_API_KEY")
[[ -n "${LLM_API_KEY:-}" ]] && \
  [[ "${LLM_API_KEY}" != sk-or-v1-* ]] && \
  echo "  ⚠️  LLM_API_KEY is set but not in OpenRouter format (sk-or-v1-...); fairness baseline assumes OpenRouter."
export LLM_BASE_URL="${LLM_BASE_URL:-https://openrouter.ai/api/v1}"
echo "  LLM_BASE_URL=${LLM_BASE_URL}"

if (( ${#missing[@]} > 0 )); then
  echo "  ❌ Missing required env vars: ${missing[*]}"
  echo "     Set them at https://claude.ai/code/routines for this routine."
  exit 1
fi
echo "::endgroup::"

echo "::group::LoCoMo dataset"
if [[ ! -f evaluation/data/locomo/locomo10.json ]]; then
  echo "  ❌ evaluation/data/locomo/locomo10.json missing"
  echo "     The dataset is committed to the repo; this is unexpected on a clean clone."
  exit 1
fi
echo "  ✅ locomo10.json present ($(wc -c < evaluation/data/locomo/locomo10.json) bytes)"
echo "::endgroup::"

echo "Setup complete."
