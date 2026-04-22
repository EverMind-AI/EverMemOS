## Auto-Bench Routine Rules

These rules apply ONLY when this session was started by the auto-bench routine. You will know because the routine prompt explicitly says so. In normal interactive sessions, ignore this section.

### Three decision rules (non-negotiable)

1. **Local memory backend only.** Only benchmark systems whose memory store/search runs in a local process or local container. Reject systems whose memory operations require a vendor SaaS endpoint.
2. **Force LLM/embedding to OpenRouter.** When integrating a candidate, rewrite its LLM and embedding config to use `LLM_BASE_URL` and `LLM_API_KEY` (which are set to OpenRouter). Never benchmark a system using its default local inference model unless that model is < 1 B parameters.
3. **Batch by RAM budget.** The cloud environment has 16 GB RAM. EverOS infra alone uses ~10 GB. If estimated total > 14 GB, run the candidate in batched chunks (split LoCoMo questions into N groups, restart docker stack between groups).

### File-level scope

Auto-bench routine sessions may modify ONLY:
- `evaluation/src/adapters/` — new adapter files only, never modify existing ones
- `evaluation/config/systems/` — new system configs only
- `evaluation/results/` — generated results
- `.auto_bench/` — internal state (seen_systems.json, scan logs)

Auto-bench routine sessions MUST NOT modify:
- Anything in `src/` (the EverOS core)
- Existing adapters or system configs
- `pyproject.toml` / `uv.lock` (no dependency changes from the routine)
- `CLAUDE.md` itself

If the routine needs a dependency that is not already in `pyproject.toml [project.optional-dependencies] evaluation-full`, abort the candidate and note this in the PR description. Do not silently add dependencies.

### Branching and PR conventions

- All routine branches MUST start with `claude/auto-bench-`. Format: `claude/auto-bench-<system-name>-YYYYMMDD`.
- Open all PRs as **draft**.
- PR title format: `[Auto-Bench] Evaluate <system-name> on <dataset>`.
- PR body MUST include, in this order:
  1. One-paragraph summary of what the system does (paraphrase from its README).
  2. Tier classification and which decision rule path was taken.
  3. Smoke test result (pass / fail / chunk-pass).
  4. Full benchmark result table comparing to top-3 already-integrated systems from `seen_systems.json`.
  5. Adapter implementation notes — anything non-obvious about how the system's interface was mapped.
  6. Footer: `Created by automated routine — session: https://claude.ai/code/${CLAUDE_CODE_REMOTE_SESSION_ID}`

### Workflow contract

For each candidate that passes discovery:

1. Clone the candidate repo into `/tmp/candidate/<name>/` (NOT into the EverOS repo working tree).
2. Read its README, identify the SDK entry point and config surface.
3. If candidate ships its own `docker-compose.yaml`, start it with a unique project name to avoid colliding with EverOS infra: `docker compose -p auto-bench-<name> up -d`.
4. Write the adapter at `evaluation/src/adapters/<name>_adapter.py` using `evermemos_adapter.py` as the structural reference (NOT mem0_adapter.py — that one targets a SaaS API and is the wrong template for our rules).
5. Write the system config at `evaluation/config/systems/<name>.yaml`. The LLM/embedding section MUST point at `${LLM_BASE_URL}` and `${LLM_API_KEY}` even if the candidate normally uses its own.
6. Run smoke first: `uv run python -m evaluation.cli --dataset locomo --system <name> --smoke`.
7. If smoke passes AND total RAM estimate ≤ 14 GB: run full LoCoMo. Skip LongMemEval unless explicitly requested — it is too token-heavy for weekly automation.
8. If smoke passes but RAM > 14 GB: run in batches (see Rule 3). The current evaluation CLI may not support `--questions-range` natively — if so, write the batched results manually and merge in the report.
9. After each candidate (pass or fail), tear down its docker stack: `docker compose -p auto-bench-<name> down -v`. THIS IS CRITICAL — without it, the next candidate will OOM.
10. Update `.auto_bench/seen_systems.json` with the new entry and commit.

### Failure modes and what to do

| Failure | Action |
|---|---|
| Candidate has no Python SDK or REST API | Add to `seen_systems.json` with `status: rejected, rejection_reason: "no programmatic interface"`. Do not open PR. |
| Candidate's `pip install` fails in the cloud env | Add to `seen_systems.json` with `status: failed, last_error: <traceback first 20 lines>`. Open PR with `[install-failed]` tag for visibility, no eval results. |
| Smoke test crashes | Open PR titled `[Auto-Bench][smoke-failed] ...` with the traceback in the body. Do not run full eval. |
| OOM during full run | Switch to batch mode. If batch mode also OOMs, abort and open PR with `[oom-batched]` tag. |
| Timeout (> 90 min total per candidate) | Abort that candidate, continue with the next one. |

### Memory and reasoning hygiene

- Do NOT speculate about benchmark numbers. Only report numbers actually emitted by `evaluation/results/<run>/eval_results.json`.
- When comparing to existing systems' historical scores, read them from `seen_systems.json` or `evaluation/results/`. Never quote a number from training data or from a paper.
- When summarizing a candidate's README in the PR, paraphrase. Do not copy more than ~15 words verbatim.
