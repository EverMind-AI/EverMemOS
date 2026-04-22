---
name: discover-memory-frameworks
description: Use this skill at the start of an auto-bench routine run to find newly released or newly trending memory frameworks for LLM agents that meet the local-deployment criteria. Triggers when the routine prompt asks to "find new memory systems", "scan for new frameworks", "check for memory framework releases", or any equivalent. The skill returns a JSON list of candidate systems that are NOT already in .auto_bench/seen_systems.json and that pass the three decision rules (local memory backend, configurable LLM/embedding, not SaaS-only).
---

# Discover Memory Frameworks

This skill is the **first step** of every auto-bench routine run. Its job is to produce a clean, deduplicated, decision-rule-filtered list of new candidate memory frameworks. Downstream skills (write-eval-adapter, run-bench) operate on this list.

## When to use

Use this skill whenever the prompt asks to find new / recent / trending memory frameworks. Do NOT use it for evaluating already-known systems — those go straight to the bench-running skill.

## Hard constraints (the three decision rules)

These come from the user's explicit decisions and are non-negotiable. A candidate that violates any one of these is `rejected`, NOT `candidate`.

**Rule 1 — Local memory backend**
The system's memory store/search operations must run in a local Python process or local Docker container. Reject any system whose store/search calls require hitting a vendor SaaS endpoint (e.g. `*.mem0.ai`, `api.getzep.com` cloud mode, `api.evermind.ai` cloud mode).

How to check: read the system's README and quickstart. Look for:
- "self-hosted" / "open source" / `docker compose up` / `pip install <pkg>` and a local `Memory()` class → likely passes Rule 1
- "Sign up for an API key" / "managed memory service" / cloud-only quickstart → likely fails Rule 1

If in doubt, look at the actual code of the quickstart example. If the quickstart `import` line gives a class that takes a base URL pointing at the vendor's domain, it fails Rule 1.

**Rule 2 — Configurable LLM and embedding**
The system's LLM and embedding model choices must be reconfigurable. The benchmark will rewrite them to point at OpenRouter via `LLM_BASE_URL` and `LLM_API_KEY`.

How to check: look for `config.json` / `settings.py` / `LLM_PROVIDER` env var / OpenAI-compatible client usage in the system code.

If LLM/embedding is hard-coded to a local model with no override, mark `unconfigurable-local-inference` and reject.

**Rule 3 — Reasonable infra footprint**
The system's local infra must plausibly fit alongside EverMemOS in 16 GB RAM. Sum estimated infra:
- EverOS baseline: 10 GB (MongoDB + Elasticsearch + Milvus + Redis)
- Candidate's required services
- Plus 2 GB safety margin

If estimated total > 14 GB, mark `oversize-infra` but DO NOT reject — flag for batch processing instead. The bench skill will handle it via chunked execution.

## Sources to scan (in priority order)

1. **arXiv recent submissions**: `cs.CL` and `cs.AI` filtered to last 7 days, search terms "memory" / "long-term memory" / "memory system" / "agent memory". Use web_search.

2. **GitHub trending**:
   - https://github.com/topics/llm-memory
   - https://github.com/topics/agent-memory
   - https://github.com/topics/long-term-memory
   - Filter by "updated in last 7 days" and "Python" language.

3. **Reference survey**: arXiv:2512.13564v2 ("Memory in the Age of LLMs") — check if a new version has been posted and diff its system inventory against `seen_systems.json`.

4. **Community signals** (lower priority, more noise):
   - r/LocalLLaMA new posts mentioning memory frameworks
   - HackerNews / Show HN tagged with memory/agent

## Filter and dedupe

For each raw candidate found:

1. Normalize the name to lowercase, strip prefixes like "the".
2. Check against `seen_systems.json` `systems[].name` and `systems[].display_name`. If matched, skip — but if matched entry has `status: "needs-revisit"`, include it.
3. Apply Rules 1, 2, 3 above using the system's README.
4. Estimate `tier` field:
   - `local-with-cloud-llm` → preferred, no rewrite needed
   - `local-with-local-llm` → acceptable, mark `requires_config_rewrite: true`
   - `saas-only` → reject

## Output format

Return a JSON object on stdout (do not narrate around it):

```json
{
  "scan_date": "2026-04-21",
  "raw_candidates_found": 7,
  "candidates": [
    {
      "name": "memoryos",
      "display_name": "MemoryOS",
      "github": "BAI-LAB/MemoryOS",
      "tier": "local-with-cloud-llm",
      "memory_backend": "local",
      "infra_required": ["sqlite"],
      "estimated_ram_gb": 1,
      "requires_config_rewrite": false,
      "discovered_via": "arXiv 2026-04-18, github trending",
      "readme_url": "https://github.com/.../README.md",
      "license": "Apache-2.0",
      "first_release_or_update": "2026-04-15"
    }
  ],
  "rejected": [
    {
      "name": "somecloudmem",
      "rejection_reason": "saas-only — quickstart requires API key from vendor",
      "rejection_rule": 1
    }
  ]
}
```

If `candidates` is empty, the routine should exit gracefully without opening any PR or sending any email.

## What NOT to do

- Do NOT open the routine's PR from inside this skill. This skill only discovers and filters.
- Do NOT clone any candidate repo from inside this skill. Cloning happens in the next skill (write-eval-adapter), and only for systems that pass all rules.
- Do NOT trust GitHub star counts or "trending" position as a signal of quality. A 50-star research framework can be more interesting than a 5k-star wrapper.
- Do NOT include MemorySparse Attention (MSA) or HyperMem — these are EverMind's own follow-up work and are tracked separately.
- Do NOT run web searches with the year hardcoded (e.g. "memory framework 2025") — use "memory framework recent" or omit the year. The current year is whatever today's date is.
