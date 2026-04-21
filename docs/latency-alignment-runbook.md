# Latency alignment — Phase 4 runbook

This document describes how to produce the canonical baseline report
that accompanies the Phase 1–3 code changes. Running it is an
operator task because it consumes external LLM APIs and a couple of
hours of wall-clock time; nothing here should be fired automatically
from CI.

## Prerequisites

- Working `.env` with `LLM_API_KEY`, `LLM_BASE_URL`, `SOPH_EMBED_URL`,
  and any other provider keys the three adapters require.
- `uv sync` has been run; `uv run pytest tests/evaluation -q` passes
  on the commit you intend to benchmark.
- Free disk in `/tmp` (or wherever `$RESULTS_ROOT` points) — each full
  1540-question run writes per-conv memcell/BM25/embedding artifacts
  plus the Layer-1 latency records. Budget ~1 GB per run.

## Step A — latency baseline (clean stat)

Goal: per-stage four-view distributions at `concurrency=1 +
retry_policy=strict_no_retry`. This is the adapter's untouched latency
with retry and backoff noise stripped — the number you quote when
comparing systems on speed alone.

```bash
bash evaluation/scripts/run_latency_baseline.sh step_a
```

What it does:
- Iterates over `evermemos`, `openclaw-native-embed`,
  `openclaw-native-noembed`.
- Each system runs 3 times over a 30-question slice of Locomo.
- `--retry-policy strict_no_retry` cascades into every
  `BenchmarkContext`; `latency_invariants.json` will fail loudly if any
  adapter retries anyway.
- `--deadline-ms 120000` caps each call at 2 min so one misbehaving
  upstream can't stall the run.
- Output: `$RESULTS_ROOT/step_a-<system>-r<1|2|3>/` each containing a
  full pipeline artifact set (benchmark_summary.json, latency_views.json,
  latency_records.json, latency_invariants.json, ...).

Runtime estimate: ≈ 15 min per system per repeat → ~2.5 h total.

## Step B — throughput baseline (production-aligned)

Goal: full-scale production-like wall time at the default harness
concurrency with each adapter's native retry strategy.

```bash
bash evaluation/scripts/run_latency_baseline.sh step_b
```

- Full 1540 questions, no smoke filter.
- `--retry-policy realistic` keeps each adapter's native retry loop
  on so 5xx / 429 / timeout signals are surfaced in the reliability
  section rather than suppressed.
- `--deadline-ms 600000` (10 min) is a hard cap per adapter call so a
  single upstream outage can't run for hours.
- Runtime estimate: 1–3 h per system depending on upstream health.

## Aggregating the results

Once both steps finish:

```bash
python evaluation/scripts/aggregate_latency_baseline.py \
    --root $RESULTS_ROOT \
    --out  docs/latency-alignment-baseline-report.md
```

The aggregator collapses every `benchmark_summary.json` under
`$RESULTS_ROOT` into one markdown table. Columns:

- Accuracy headline (Acc / content_overlap@5 / F1 / BLEU-1 /
  retry_policy).
- Per-stage `wall_ms` distributions for `add`, `search`, `answer`,
  and the derived `e2e_query_ms`.
- For each stage: `realistic` p50/p95, `clean` p50/p95,
  `first_attempt` p50, `successful_attempt` p50 side-by-side —
  these are the four views from docs/latency-alignment.md.
- Reliability: retry%, failed%.
- Invariant-check summary (error / warning counts, grouped by code).

## Reading the output

Interpretation cheat sheet:

- **realistic wall_ms** — what a user feels. Worse when upstreams
  misbehave.
- **clean wall_ms** — filter out retried/failed calls. "System speed
  in a good state." Expect clean p95 ≪ realistic p95 if the network
  was flaky during the run.
- **first_attempt** — how fast the adapter is when it issues exactly
  one call. Unaffected by retry cost. Use this when comparing
  algorithmic approaches.
- **successful_attempt** — how fast a successful attempt was,
  regardless of prior failures. When this equals `first_attempt`,
  first-attempt success is the norm; when they diverge, the adapter
  is benefiting from warmup / caching across retries.

- **retry% / failed%** non-zero under `realistic` policy is expected
  upstream reality; non-zero under `strict_no_retry` is a contract
  violation and latency_invariants.json will carry a
  `strict_policy` error item.
- **latency_invariants.count** should be zero on a healthy run. Any
  `wall_mismatch` warnings indicate an adapter that adopts
  `record_attempt` incompletely (it reported some attempts but missed
  others, or inflated the wait buckets).

## Publishing

Commit the generated `docs/latency-alignment-baseline-report.md` on
the same branch as the Phase-1-3 code, so reviewers can tie the
numbers to the measurement changes that produced them.
