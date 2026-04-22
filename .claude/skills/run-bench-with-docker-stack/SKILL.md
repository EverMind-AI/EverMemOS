---
name: run-bench-with-docker-stack
description: Use after write-eval-adapter has produced an adapter + config + registry entry for a candidate memory framework. This skill runs the LoCoMo benchmark against that one candidate, orchestrating the EverOS docker infra and the candidate's own docker stack (if any), handling 16 GB RAM budget via batch splits, and emitting a single merged report and updated seen_systems.json entry. Triggers when the routine prompt says "run bench for <system>", "benchmark <system>", or chains into this step after adapter is in place. Does NOT open the PR — that is the main prompt's job.
---

# Run Bench With Docker Stack

This skill is the **third step** of every auto-bench routine run. It takes ONE candidate whose adapter is already written and registered, executes the LoCoMo evaluation inside the cloud container's 16 GB RAM budget, and leaves results ready for the routine's PR step.

## When to use

- After `write-eval-adapter` has written `evaluation/src/adapters/<name>_adapter.py`, `evaluation/config/systems/<name>.yaml`, and added the registry line.
- Once per candidate. If the routine has 3 candidates, call this skill 3 times — with a full docker teardown between each.
- Do NOT use this skill if the adapter file is missing or the config is missing — fail loudly instead.

## Preconditions you MUST verify before starting

Run each of these and abort with a clear message if any fails:

```bash
# 1. Adapter file exists
test -f evaluation/src/adapters/<name>_adapter.py

# 2. System config exists
test -f evaluation/config/systems/<name>.yaml

# 3. Registry has the new entry
python -c "from evaluation.src.adapters.registry import list_adapters; assert '<name>' in list_adapters()"

# 4. Required env vars are present
: "${LLM_API_KEY:?LLM_API_KEY (OpenRouter key sk-or-v1-...) is required}"
: "${LLM_BASE_URL:=https://openrouter.ai/api/v1}"

# 5. LoCoMo data is present
test -f evaluation/data/locomo10.json || {
  echo "LoCoMo data missing — run the data-download script"
  exit 1
}
```

## RAM budget and batching algorithm

**Budget facts (verified):**
- Cloud container: 16 GB RAM total.
- EverOS infra baseline (MongoDB + Elasticsearch + Milvus + Redis): ~10 GB when all services are running.
- Safety margin: 2 GB (Python driver, docker daemon overhead, OS).
- Usable budget for candidate: **4 GB** if EverOS stack must stay up, **14 GB** if the candidate does not need any of EverOS's services.

**Decision tree for candidate start-up:**

1. Read `estimated_ram_gb` for this candidate from `.auto_bench/seen_systems.json`. If missing, default to 3.
2. Does the adapter import anything from `memory_layer.*` or `agentic_layer.*`? (External-call adapters should NOT — this should be NO.)
   - If YES: the adapter is mis-classified; abort and escalate to human.
   - If NO (expected): EverOS docker stack is NOT required for this candidate. Stop it before benchmarking:

     ```bash
     docker compose -f docker-compose.yaml down
     ```

     Now the full 14 GB is available for the candidate.

3. If the candidate ships its own `docker-compose.yaml`, start it with a unique project name:

   ```bash
   cd /tmp/candidate/<name>/
   docker compose -p auto-bench-<name> up -d
   # Wait for health checks
   docker compose -p auto-bench-<name> ps
   ```

   Project-name isolation prevents candidate services with generic names (`postgres`, `redis`, `chroma`, etc.) from colliding with EverOS's own stack IF you later decide to run both. For this routine we always stop EverOS first, so this is belt-and-suspenders.

4. If `estimated_ram_gb > 12`, set `BATCH_SIZE=2` (5 batches × 2 convs each). Else if `> 6`, set `BATCH_SIZE=5` (2 batches × 5 convs). Else `BATCH_SIZE=10` (single run).

## Output-path contract (READ FIRST — harness gotchas)

These three facts come from `evaluation/cli.py` and `evaluation/src/core/pipeline.py` and dictate the whole batching design:

1. **`--to-conv` is EXCLUSIVE.** `--from-conv 0 --to-conv 5` processes conversations 0..4 (five conversations). Source: `cli.py:107`, `pipeline.py:96,109,617`.
2. **Default output dir = `results/{dataset}-{system}[-{run_name}]`.** Passing `--run-name foo` does NOT produce `results/foo/`. Source: `cli.py:211-223`. Always pass `--output-dir` explicitly so the same path is readable and isolatable.
3. **Pipeline loads checkpoint from output_dir and skips completed stages.** Re-running the CLI with the same `--output-dir` after any stage is marked complete causes that stage to be silently skipped (`pipeline.py:218-237,252`). The CLI exposes no flag to disable checkpoints. Therefore two invocations that must do NEW work MUST use DIFFERENT output directories.

Everything below is written to respect these three facts.

## Pick paths first, then run

```bash
cd "$CLAUDE_PROJECT_DIR"
TS=$(date +%Y%m%d-%H%M%S)
SMOKE_DIR="evaluation/results/auto-bench-<name>-smoke-$TS"
FULL_BASE="evaluation/results/auto-bench-<name>-full-$TS"
```

`$FULL_BASE` is a directory that contains one subdirectory per batch (or one subdirectory for the single-batch case) plus a merged summary JSON. Never write anything above `$FULL_BASE` from this skill.

## Smoke test FIRST (always)

Before any full run, prove the adapter is wired up. Explicit `--output-dir` makes the output path knowable and keeps the smoke run out of the full-run namespace:

```bash
uv run python -m evaluation.cli \
  --dataset locomo \
  --system <name> \
  --smoke \
  --smoke-messages 20 \
  --smoke-questions 5 \
  --output-dir "$SMOKE_DIR" \
  --clean-groups
```

Interpret the smoke result:

| Outcome | Action |
|---|---|
| Exits 0, `"$SMOKE_DIR/eval_results.json"` exists and is non-empty | Proceed to full run |
| ImportError on candidate package | Stop. Write failure row to `seen_systems.json` with `status: failed`, `rejection_reason: "missing dep — not in evaluation-full"`. No PR; report to routine prompt. |
| HTTP error from candidate service | Inspect docker logs: `docker compose -p auto-bench-<name> logs --tail 100`. If service never became healthy, mark `status: failed`, `rejection_reason: "service unhealthy on boot"`. |
| `results` empty for all queries | Likely search method name wrong OR async indexing not awaited. Try adding `post_add_wait_seconds: 60` to config, re-run smoke ONCE (to a FRESH `$SMOKE_DIR` — don't reuse). If still empty: `status: failed`, `rejection_reason: "empty retrieval"`. |
| OOM (container killed) | Stop. Mark `tier: oversize-infra`. No PR from this run; routine's main prompt will record this for the next iteration. |
| Exits 0 but score is 0 for all 5 questions | Wire is correct but scoring is zero — proceed to full run anyway; humans will investigate via PR `[zero-score]` tag. |

## Full run

Only reachable if smoke passed. In both single- and multi-batch paths each CLI invocation gets its OWN `--output-dir` to avoid the checkpoint-skip bug described above.

**Single-batch path** (when `BATCH_SIZE == 10`):

```bash
BATCH_DIR="$FULL_BASE/all"
mkdir -p "$FULL_BASE"

uv run python -m evaluation.cli \
  --dataset locomo \
  --system <name> \
  --output-dir "$BATCH_DIR" \
  --clean-groups
```

The per-batch `eval_results.json` lands at `"$BATCH_DIR/eval_results.json"`.

**Multi-batch path** (when `BATCH_SIZE < 10`):

LoCoMo has 10 conversations (indices 0..9). Slice into contiguous ranges using EXCLUSIVE upper bounds (to match the CLI's `--to-conv` semantics) and give each batch a distinct output subdirectory (to avoid the checkpoint carry-over):

```bash
NCONVS=10
mkdir -p "$FULL_BASE"

# Enumerate exclusive ranges: (0, BATCH_SIZE), (BATCH_SIZE, 2*BATCH_SIZE), ...
BATCH_IDX=0
for START in $(seq 0 "$BATCH_SIZE" $((NCONVS - 1))); do
  END=$(( START + BATCH_SIZE ))
  [ $END -gt $NCONVS ] && END=$NCONVS

  BATCH_DIR="$FULL_BASE/batch-$BATCH_IDX-convs-$START-$((END - 1))"
  echo "=== Batch $BATCH_IDX: convs [$START, $END) → $BATCH_DIR ==="

  # Restart candidate stack to free transient memory between batches
  if [ -d /tmp/candidate/<name> ]; then
    docker compose -p auto-bench-<name> -f /tmp/candidate/<name>/docker-compose.yaml restart
    sleep 20  # let the stack settle
  fi

  uv run python -m evaluation.cli \
    --dataset locomo \
    --system <name> \
    --from-conv "$START" \
    --to-conv "$END" \
    --output-dir "$BATCH_DIR" \
    --clean-groups

  BATCH_IDX=$((BATCH_IDX + 1))
done
```

Key invariants this loop enforces (and why the previous version was wrong):

- `END = START + BATCH_SIZE` (NOT `START + BATCH_SIZE - 1`). The CLI treats `--to-conv` as exclusive, so the previous `-1` dropped the last conversation in every batch — a silent coverage bug.
- Each batch writes to a distinct `$BATCH_DIR`, so the next batch's Pipeline reads an empty checkpoint and actually runs the stages instead of skipping them.
- `$FULL_BASE` contains only batch subdirs; the merged summary goes one level up from each batch (see below), never overwriting a batch's own files.

## Merge and completeness check

Single-batch: read `"$FULL_BASE/all/eval_results.json"` directly.

Multi-batch: merge the per-batch JSONs, then assert every expected conversation_id is present. Do the merge with a small Python block inside the skill — shell JSON manipulation is not worth the risk. Pass `$FULL_BASE` as an argv, not by heredoc interpolation (quoted heredoc deliberately disables substitution so the Python body stays literal):

```bash
uv run python - "$FULL_BASE" <<'PY'
import json
import sys
from pathlib import Path

full_base = Path(sys.argv[1])
batch_dirs = sorted(p for p in full_base.iterdir() if p.is_dir() and p.name.startswith("batch-"))

if not batch_dirs:  # single-batch fallback
    batch_dirs = [full_base / "all"]

merged = {"per_conversation": {}, "batch_files": []}
for bd in batch_dirs:
    f = bd / "eval_results.json"
    if not f.exists():
        raise SystemExit(f"missing batch result: {f}")
    data = json.loads(f.read_text())
    merged["batch_files"].append(str(f))
    # The eval_results.json schema is harness-defined; do not invent field names here.
    # Record the raw dict per batch and let the PR step reason over it.
    merged.setdefault("batches", []).append({"batch_dir": str(bd), "eval": data})

# Completeness check: we expect exactly 10 LoCoMo conversations covered.
# Derive covered conv IDs from the loaded dataset slice used by each batch.
# The harness-side per_conversation_metrics key is not guaranteed; rely on the
# batch_dir names which encode the inclusive range we asked for.
covered = set()
for bd in batch_dirs:
    if bd.name == "all":
        covered |= set(range(10))
    else:
        # name format: batch-<i>-convs-<lo>-<hi>  (hi inclusive, per our mkdir)
        parts = bd.name.split("-")
        lo, hi = int(parts[-2]), int(parts[-1])
        covered |= set(range(lo, hi + 1))

expected = set(range(10))
missing = sorted(expected - covered)
if missing:
    raise SystemExit(f"COVERAGE GAP: missing LoCoMo conversation indices {missing}. "
                     "Do NOT publish this result.")

(full_base / "merged_summary.json").write_text(json.dumps(merged, indent=2))
print(f"Merged {len(batch_dirs)} batch(es); coverage complete (0..9).")
PY
```

The completeness assertion is non-negotiable: if a batch silently dropped or skipped a conversation, the routine MUST abort here and open the PR with `[coverage-gap]` tag, with NO metrics in the body.

After the merge passes, the PR step reads `"$FULL_BASE/merged_summary.json"` to compose the metrics table. Do not construct summaries from log output.

Then compare against the top-3 already-integrated systems from `.auto_bench/seen_systems.json` (look for entries with `status: integrated` and a `last_benchmark` block). If `seen_systems.json` has no benchmark history for comparison yet, note that explicitly in the report body instead of inventing numbers.

## Update seen_systems.json

After the run completes (pass OR fail), append/update this candidate's entry:

```json
{
  "name": "<name>",
  "display_name": "<SystemName>",
  "github": "<owner>/<repo>",
  "first_seen": "<YYYY-MM-DD — today>",
  "status": "candidate|failed",
  "tier": "local-with-cloud-llm",
  "memory_backend": "local",
  "adapter_path": "evaluation/src/adapters/<name>_adapter.py",
  "config_path": "evaluation/config/systems/<name>.yaml",
  "infra_required": ["..."],
  "estimated_ram_gb": <int>,
  "last_run": {
    "date": "<YYYY-MM-DD>",
    "run_name": "<RUN_NAME>",
    "smoke_result": "pass|fail",
    "full_result": "pass|partial|fail",
    "batch_mode": true|false,
    "overall_score": <float or null>,
    "notes": "<1–2 sentences>"
  }
}
```

Do NOT overwrite the file blindly — read it, update the single entry, write it back with 2-space indent.

## Teardown (ALWAYS run at the end, even on failure)

This is CRITICAL. Without teardown, the next routine run will OOM on startup.

```bash
# Stop candidate stack
if [ -d /tmp/candidate/<name> ]; then
  docker compose -p auto-bench-<name> -f /tmp/candidate/<name>/docker-compose.yaml down -v
fi

# Clean any orphaned candidate volumes (project-scoped only)
docker volume ls --filter "label=com.docker.compose.project=auto-bench-<name>" -q | xargs -r docker volume rm

# If EverOS stack was stopped earlier and the next step needs it, restart here.
# For the auto-bench routine, leaving it stopped is fine — the next routine run will bring it up if needed.
```

Verify teardown:

```bash
docker ps --filter "label=com.docker.compose.project=auto-bench-<name>" -q | wc -l
# Expect: 0
```

## Timeout budget

The routine's overall wall-clock budget is ~90 minutes per candidate. Hard ceilings:

- Smoke test: 10 minutes max. Kill and mark `status: failed` if exceeded.
- Full run (single batch): 50 minutes max.
- Full run (multi-batch): 75 minutes total max, even if some batches finished. Abort remaining batches and mark `full_result: partial`.
- Teardown: 5 minutes max. If containers won't stop, `docker kill` and report in PR body.

Use `timeout` wrappers:

```bash
timeout 600 uv run python -m evaluation.cli ... --smoke --output-dir "$SMOKE_DIR" ...
timeout 3000 uv run python -m evaluation.cli ... --from-conv "$START" --to-conv "$END" --output-dir "$BATCH_DIR" ...
```

`$START` is inclusive, `$END` is exclusive — these match the CLI's semantics directly. Do not subtract one from `$END`.

## What NOT to do in this skill

- Do NOT write the PR body here. The routine's main prompt composes the PR.
- Do NOT git commit — the main prompt batches commits after teardown.
- Do NOT start EverOS's docker-compose.yaml for a routine candidate — candidates are local black-box integrations that don't need MongoDB/ES/Milvus/Redis. It's wasted RAM. (The only case EverOS infra is needed is for the native `evermemos` adapter, which is never a routine candidate.)
- Do NOT pull candidate docker images on-the-fly if the setup script already did it — rely on the setup-script-cached images to avoid network blips.
- Do NOT modify `evaluation/cli.py` or any harness file. If the CLI lacks a flag the routine needs, abort and file a note in the PR body instead.
- Do NOT delete `evaluation/results/` — those are sources of truth.
- Do NOT quote scores from training data, the candidate's paper, or log output. Only quote numbers that came from the `eval_results.json` you just wrote.
