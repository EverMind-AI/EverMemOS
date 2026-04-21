"""
Collapse many per-run benchmark_summary.json files into one comparison
markdown table. Consumes the layout produced by
``evaluation/scripts/run_latency_baseline.sh``.

Usage:
    python evaluation/scripts/aggregate_latency_baseline.py \
        --root /tmp/latency-baseline-... \
        --out docs/latency-alignment-baseline-report.md
"""
from __future__ import annotations

import argparse
import json
import statistics as _st
from pathlib import Path
from typing import Dict, List, Optional


def _load(dir_: Path) -> Optional[dict]:
    p = dir_ / "benchmark_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"


def _fmt_ms(v: Optional[float]) -> str:
    return f"{v:,.0f}" if isinstance(v, (int, float)) else "n/a"


def _pick(view: Optional[dict], stat: str) -> Optional[float]:
    if not view:
        return None
    return view.get(stat)


def _stage_block(stage: Optional[dict]) -> Dict[str, str]:
    """Produce a flat dict of printable strings for one stage's views."""
    if not stage:
        return {}
    wall = stage.get("wall_ms") or {}
    rel = stage.get("reliability") or {}
    return {
        "n": str(stage.get("n_calls", "n/a")),
        "realistic_p50": _fmt_ms(_pick(wall.get("realistic"), "p50")),
        "realistic_p95": _fmt_ms(_pick(wall.get("realistic"), "p95")),
        "clean_p50": _fmt_ms(_pick(wall.get("clean"), "p50")),
        "clean_p95": _fmt_ms(_pick(wall.get("clean"), "p95")),
        "first_p50": _fmt_ms(_pick(wall.get("first_attempt"), "p50")),
        "succ_p50": _fmt_ms(_pick(wall.get("successful_attempt"), "p50")),
        "retry_rate": _fmt_pct(rel.get("retry_rate")),
        "failed_rate": _fmt_pct(rel.get("failed_rate")),
    }


def render(runs: Dict[str, dict]) -> str:
    """runs: {label: benchmark_summary dict}. Returns markdown."""
    lines: List[str] = []

    lines.append("# Latency alignment — baseline report\n")
    lines.append(
        "Each row is one pipeline run. Latency columns are harness-measured "
        "at the adapter boundary (see docs/latency-alignment.md).\n"
    )

    lines.append("## Accuracy / retrieval headline\n")
    lines.append(
        "| Run | Acc | content_overlap@5 | F1 | BLEU-1 | retry_policy |"
    )
    lines.append("|---|---|---|---|---|---|")
    for label, s in runs.items():
        a = s.get("answer_level") or {}
        r = s.get("retrieval_level") or {}
        co = r.get("content_overlap_at_5")
        lines.append(
            f"| {label} | {_fmt_pct(a.get('accuracy'))} | "
            f"{(co if isinstance(co, (int, float)) else 'n/a') if co is None else f'{co:.3f}'} | "
            f"{a.get('f1_mean') or 'n/a':.3f} | "
            f"{a.get('bleu1_mean') or 'n/a':.3f} | "
            f"{s.get('retry_policy','?')} |"
        )
    lines.append("")

    for stage in ("add", "search", "answer", "e2e_query_ms"):
        lines.append(f"## {stage} wall_ms (ms)\n")
        lines.append(
            "| Run | n | realistic p50 / p95 | clean p50 / p95 | first_attempt p50 | successful_attempt p50 | retry% | failed% |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|"
        )
        for label, s in runs.items():
            lat = s.get("latency") or {}
            blk = _stage_block(lat.get(stage))
            if not blk:
                lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {label} | {blk['n']} | "
                f"{blk['realistic_p50']} / {blk['realistic_p95']} | "
                f"{blk['clean_p50']} / {blk['clean_p95']} | "
                f"{blk['first_p50']} | "
                f"{blk['succ_p50']} | "
                f"{blk['retry_rate']} | {blk['failed_rate']} |"
            )
        lines.append("")

    lines.append("## Invariant check\n")
    lines.append("| Run | errors | warnings | codes |")
    lines.append("|---|---|---|---|")
    for label, s in runs.items():
        inv = s.get("latency_invariants") or {}
        by_sev = inv.get("by_severity") or {}
        lines.append(
            f"| {label} | {by_sev.get('error', 0)} | "
            f"{by_sev.get('warning', 0)} | {inv.get('by_code', {})} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Directory holding step_a-*/step_b-* subdirs.")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output markdown path.")
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"root {args.root} does not exist")

    runs: Dict[str, dict] = {}
    for sub in sorted(args.root.iterdir()):
        if not sub.is_dir():
            continue
        summary = _load(sub)
        if summary is None:
            continue
        runs[sub.name] = summary

    if not runs:
        raise SystemExit(f"no benchmark_summary.json under {args.root}")

    md = render(runs)
    args.out.write_text(md)
    print(f"Wrote {args.out} ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
