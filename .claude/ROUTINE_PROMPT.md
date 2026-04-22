# Routine main prompt — Auto-Bench for DuffyCoder/EverOS

> Paste the block below into the "Prompt" field when creating the routine at
> https://claude.ai/code/routines. Detail lives in the three skills
> (`discover-memory-frameworks`, `write-eval-adapter`,
> `run-bench-with-docker-stack`) and in `CLAUDE.md` — keep this prompt short.

---

## Prompt (paste into routine)

You are the auto-bench routine for DuffyCoder/EverOS. Follow the rules in
`CLAUDE.md § Auto-Bench Routine Rules`. All detailed steps live in skills — do
not re-derive the work.

Run, in order:

1. **Discover.** Invoke the `discover-memory-frameworks` skill. It returns a
   JSON list of candidates. If `candidates` is empty, exit immediately — no
   PR, no email.

2. **For each candidate in the list** (sequentially, not in parallel — skills
   touch `registry.py`):
   a. Checkout a fresh branch: `claude/auto-bench-<name>-$(date +%Y%m%d)`.
   b. Invoke the `write-eval-adapter` skill with the candidate's record.
   c. Invoke the `run-bench-with-docker-stack` skill. It produces
      `evaluation/results/<run-name>/eval_results.json` and updates
      `.auto_bench/seen_systems.json`.
   d. Commit: adapter file, config file, registry edit, `seen_systems.json`.
      Commit message: `[Auto-Bench] Add <name> adapter — LoCoMo <pass|fail>`.
      Push branch.
   e. Open DRAFT PR per `CLAUDE.md § Branching and PR conventions`. Title
      `[Auto-Bench] Evaluate <name> on LoCoMo` plus any failure tag. Body
      template is in the addendum — fill from `eval_results.json` only.
   f. Teardown already happened inside the run-bench skill. Verify no
      `auto-bench-<name>-*` containers remain before moving on.

3. **Notify.** After all candidates are processed, send ONE Gmail via the
   Gmail connector:
     - To: the routine owner's configured email.
     - Subject: `Auto-Bench weekly: <N_pass>/<N_total> candidates passed`.
     - Body: one bullet per candidate with PR URL and headline metric.

## Non-negotiables (abort if violated)

- Only benchmark systems with a local memory backend (Rule 1 in the skill).
- LLM/embedding config MUST be rewritten to `${LLM_API_KEY}` /
  `${LLM_BASE_URL}` (OpenRouter) before smoke (Rule 2).
- If `estimated_ram_gb > 14` after stopping EverOS stack, run LoCoMo in
  batches via `--from-conv`/`--to-conv` (Rule 3).
- Do NOT add dependencies. If a candidate needs a missing pip package, open
  the PR with `[install-failed]` tag and no eval results.
- Do NOT modify anything outside `evaluation/src/adapters/`,
  `evaluation/config/systems/`, `evaluation/results/`, `.auto_bench/`.
- Never invent benchmark numbers. Only quote what you read from
  `eval_results.json`.

## First-run dry-run instruction

For the FIRST routine run (this paragraph should be deleted once validated):
**DRY RUN — run discover + write-adapter steps, but skip `run-bench` and skip
PR creation. Instead, print the adapter file diff and the system YAML to the
session log. Send no email.**

## Failure behavior

If any step throws, record the failure in `.auto_bench/seen_systems.json`
against that candidate (`status: failed`, `last_error: <first 20 lines>`),
continue to the next candidate, and include the failure in the Gmail summary.
Do not abort the whole routine on one candidate's failure.

## Session context for humans

The Gmail body and the PR footer both include
`session: https://claude.ai/code/${CLAUDE_CODE_REMOTE_SESSION_ID}` so humans
can replay the run.
