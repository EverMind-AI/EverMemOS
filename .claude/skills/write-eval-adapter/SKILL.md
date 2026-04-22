---
name: write-eval-adapter
description: Use after discover-memory-frameworks has produced a candidate list. For each candidate (one at a time), this skill writes an adapter at `evaluation/src/adapters/<name>_adapter.py`, a system config at `evaluation/config/systems/<name>.yaml`, and a one-line addition to `evaluation/src/adapters/registry.py`. The adapter treats the candidate as a black-box local service (in-process SDK or localhost HTTP). Triggers when the routine prompt says "write adapter for <system>", "integrate <system>", or chains into this step after discovery. Does NOT run smoke tests or open PRs — that is the next skill's job.
---

# Write Eval Adapter

This skill is the **second step** of every auto-bench routine run. Its job is to produce a working, registered adapter for ONE candidate memory framework at a time.

## Naming clarification (important)

The base class is called `OnlineAPIAdapter` for historical reasons. Do not read it as "cloud SaaS adapter". Its actual role is **black-box template method base class**: it provides three pieces of generic evaluation scaffolding that any system — local-SDK, local-HTTP, or SaaS — can share:

1. Conversation-level concurrency control (`num_workers` semaphore).
2. Dual-perspective handling for LoCoMo's `speaker_a` / `speaker_b`.
3. `answer()` implementation built on `LLMProvider` (OpenRouter).

The local-vs-SaaS distinction is enforced at discovery time (Rule 1 rejects SaaS candidates), not at the base-class level. Proof: `evermemos_api_adapter.py` is a local-HTTP adapter (`http://localhost:1995`) and inherits `OnlineAPIAdapter` — it is the canonical reference for this skill.

**The structural reference for every new adapter is `evermemos_api_adapter.py`.** It shows the local-HTTP variant. For local-SDK candidates, use the same template-method overrides but construct the candidate's Python class in `__init__` instead of an `aiohttp.ClientSession`. Do NOT use `mem0_adapter.py` as a template — it targets Mem0's SaaS endpoint, which our Rule 1 already rejected.

The alternative (inherit `BaseAdapter` directly) would drop the concurrency/perspective/answer scaffolding and force every new adapter to re-implement ~400 lines of harness plumbing. Not worth it.

## When to use

- After `discover-memory-frameworks` returns a non-empty `candidates` list.
- For each candidate: invoke this skill once. Sequential, not parallel — each call touches `registry.py` and can race.
- Do NOT use this skill on `evermemos` or any `status: integrated` system in `seen_systems.json`.
- Do NOT use this skill to modify an existing adapter — this skill only creates new files.

## Hard rules (the non-negotiables)

**Rule A — Black-box local integration.** The new adapter MUST inherit from `OnlineAPIAdapter` (from `evaluation.src.adapters.online_base`) — see the naming clarification above. Do NOT inherit from `BaseAdapter` directly. Do NOT import anything from `src/memory_layer/` or `src/agentic_layer/` — those are EverMemOS internals reserved for the privileged `evermemos_adapter.py` path. The candidate must be treated as a black box: call its public SDK or HTTP endpoints only, never reach into EverMemOS primitives.

**Rule B — Force LLM/embedding to OpenRouter.** The system config's `llm:` block MUST read `api_key: "${LLM_API_KEY}"` and `base_url: "${LLM_BASE_URL:https://openrouter.ai/api/v1}"` and `model: "openai/gpt-4.1-mini"` (the fairness baseline). If the candidate also has its own internal LLM/embedding config (e.g. writes calls to Ollama or a bundled local model), the adapter's `__init__` MUST override those at runtime using `os.environ["LLM_BASE_URL"]` and `os.environ["LLM_API_KEY"]` before constructing the candidate's client.

**Rule C — No dependency changes.** If the candidate requires a new pip package that is not already in `pyproject.toml [project.optional-dependencies] evaluation-full`, STOP. Do not edit `pyproject.toml`. Write the adapter file skeleton anyway but mark the import with:

```python
try:
    from candidatepkg import CandidateClient
except ImportError:
    raise ImportError(
        "candidatepkg not installed. Add to pyproject.toml evaluation-full group "
        "and open a separate PR before running this adapter."
    )
```

Then the next skill (run-bench) will see the import error and emit a `[install-failed]` PR.

## Adapter file template

Write to `evaluation/src/adapters/<name>_adapter.py` (flat file, not subdir — the repo convention is flat files; only `evermemos/` and `openclaw/` are subdirs because they are privileged re-implementation paths).

Use this skeleton. Fill in the four clearly-marked TODO blocks.

```python
"""
<SystemName> Adapter — local black-box integration, OpenRouter-rewritten LLM.

Auto-bench routine notes:
- Inherits OnlineAPIAdapter template method.
- <One paragraph, paraphrased from candidate README. No more than 15 words verbatim.>
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console

from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult


@register_adapter("<name>")
class <SystemName>Adapter(OnlineAPIAdapter):
    """
    <SystemName> adapter (local deployment, black-box integration).

    Config example:
    ```yaml
    adapter: "<name>"
    base_url: "http://localhost:<port>"  # or SDK-only: omit
    api_key: ""
    num_workers: 5
    llm:
      model: "openai/gpt-4.1-mini"
      api_key: "${LLM_API_KEY}"
      base_url: "${LLM_BASE_URL:https://openrouter.ai/api/v1}"
    ```
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        # --- TODO 1: force-rewrite candidate's LLM/embedding env (Rule B) ---
        # If the candidate reads LLM config from env at import time, set env vars
        # BEFORE importing / constructing its client. Example:
        os.environ.setdefault("OPENAI_BASE_URL", os.environ.get("LLM_BASE_URL", ""))
        os.environ.setdefault("OPENAI_API_KEY", os.environ.get("LLM_API_KEY", ""))

        # --- TODO 2: construct the candidate client (Rule A, Rule C) ---
        try:
            from candidatepkg import CandidateClient  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"<name> not installed: {e}. Not in evaluation-full — add dependency "
                "in a separate PR before running this adapter."
            ) from e

        self.base_url = str(config.get("base_url", "") or "").rstrip("/")
        self.api_key = str(config.get("api_key", "") or "")
        self.max_retries = int(config.get("max_retries", 3))
        self.request_interval = float(config.get("request_interval", 0.0))

        self.client = CandidateClient(
            base_url=self.base_url or None,
            api_key=self.api_key or None,
            # If the client accepts llm overrides, wire them from config["llm"] here.
        )
        self.console = Console()
        print(f"   <SystemName> client constructed (base_url={self.base_url or 'sdk-local'})")

    # ---- Rule: most candidates do not support dual-perspective group chat. ----
    #           Override only if the candidate explicitly supports multiple user_ids.
    def _need_dual_perspective(self, speaker_a: str, speaker_b: str) -> bool:
        return super()._need_dual_perspective(speaker_a, speaker_b)

    # --- TODO 3: ingest (Stage 1 — add) ---
    async def _add_user_messages(
        self,
        conv: Conversation,
        messages: List[Dict[str, Any]],
        speaker: str,
        **kwargs: Any,
    ) -> Any:
        user_id = self._extract_user_id(conv, speaker=speaker)
        progress = kwargs.get("progress")
        task_id = kwargs.get("task_id")

        for attempt in range(self.max_retries):
            try:
                # Call the candidate's ingest API. Shape guesses:
                #   self.client.add(messages=messages, user_id=user_id, ...)
                # or looped per-message:
                #   for m in messages: self.client.ingest(user_id, m["content"], ...)
                # Use whichever matches the candidate's public API.
                await asyncio.to_thread(
                    self.client.add,          # TODO: replace with real method name
                    messages=messages,
                    user_id=user_id,
                )
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        if progress is not None and task_id is not None:
            progress.update(task_id, advance=len(messages))
        if self.request_interval > 0:
            await asyncio.sleep(self.request_interval)
        return None

    # --- TODO 4: retrieve (Stage 2 — search) ---
    async def _search_single_user(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        top_k: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        raw = await asyncio.to_thread(
            self.client.search,           # TODO: replace with real method name
            query=query,
            user_id=user_id,
            top_k=top_k,
        )

        # Normalize to standard format required by OnlineAPIAdapter
        out: List[Dict[str, Any]] = []
        for item in (raw or []):
            content = item.get("text") or item.get("memory") or item.get("content") or ""
            ts = item.get("timestamp") or item.get("created_at") or ""
            out.append({
                "content": f"{ts}: {content}".strip(": ").strip() if ts else content,
                "score": float(item.get("score", 0.0)),
                "user_id": user_id,
                "metadata": {"raw": item},
            })
        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out[: int(top_k)]

    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs: Any,
    ) -> SearchResult:
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results[: int(top_k)],
            retrieval_metadata={
                "system": "<name>",
                "top_k": int(top_k),
                "dual_perspective": False,
                "user_ids": [user_id],
            },
        )

    def _build_dual_search_result(
        self,
        query: str,
        conversation_id: str,
        all_results: List[Dict[str, Any]],
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]],
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int,
        **kwargs: Any,
    ) -> SearchResult:
        # Reuse default template from prompts.yaml
        speaker_a_text = "\n".join(r["content"] for r in results_a) if results_a else "(No memories found)"
        speaker_b_text = "\n".join(r["content"] for r in results_b) if results_b else "(No memories found)"
        template = self._prompts["online_api"].get("templates", {}).get("default", "")
        formatted = template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_text,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_text,
        )
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=all_results,
            retrieval_metadata={
                "system": "<name>",
                "top_k": int(top_k),
                "dual_perspective": True,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "formatted_context": formatted,
            },
        )

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "<SystemName>",
            "type": "online_api",
            "adapter": "<SystemName>Adapter",
        }
```

### Two skeleton variants — pick at TODO 2

**Variant A — Python SDK (in-process).** The candidate exposes a class you construct with kwargs. Pattern: `CandidateClient(...)` with in-memory storage, OR a client that talks to localhost via its own transport. Use `asyncio.to_thread(self.client.method, ...)` in `_add_user_messages` and `_search_single_user` to bridge sync SDKs. Example reference: the Mem0 local mode pattern (if it were implemented) or any `Memory()` class.

**Variant B — HTTP API.** The candidate ships a docker-compose file with a REST server on `localhost:<port>`. Pattern: use `aiohttp.ClientSession` directly (see `evermemos_api_adapter.py` at `_request_json_with_retry`). Do NOT use `requests` — this codebase is async. Pass `base_url` and `api_key` through config.

## System config template

Write to `evaluation/config/systems/<name>.yaml`:

```yaml
# <SystemName> System Configuration (auto-generated by write-eval-adapter skill)

name: "<name>"
version: "1.0"
description: "<SystemName> — <one-line paraphrased purpose>"

adapter: "<name>"

# <SystemName>-specific configuration
base_url: "http://localhost:<port>"  # only for HTTP candidates; remove for SDK-only
api_key: ""                           # most local candidates don't require auth
max_retries: 3
request_interval: 0.0

# Concurrency (conversation-level; keep low until smoke test passes)
num_workers: 5

# Search configuration
search:
  top_k: 20

# LLM configuration — FORCED to OpenRouter (Rule B, non-negotiable)
llm:
  provider: "openai"
  model: "openai/gpt-4.1-mini"
  api_key: "${LLM_API_KEY}"
  base_url: "${LLM_BASE_URL:https://openrouter.ai/api/v1}"
  temperature: 0
  max_tokens: 32768

answer:
  max_retries: 3
```

If the candidate needs additional env vars (e.g. a license key required just for startup, not for inference), add them under a top-level `env:` key for documentation only. They must be actual env vars on the cloud container — do NOT hardcode values.

## Register the adapter

Add ONE line to `evaluation/src/adapters/registry.py` inside `_ADAPTER_MODULES`:

```python
"<name>": "evaluation.src.adapters.<name>_adapter",
```

Place it alongside the other `# Online API systems` comment block. Do not reorder existing entries. Do not remove any entry.

## Batching contract for Rule 3 (RAM-aware split)

The evaluation CLI supports `--from-conv I --to-conv J` natively on LoCoMo. The adapter does NOT need to do anything special for batching — the CLI's `--clean-groups` + `--from-conv`/`--to-conv` pair is sufficient, provided:

- The adapter's `add()` is idempotent within a conversation (safe to re-run a conv on the same `user_id` if a batch is retried). Most external APIs are idempotent by message ID; if the candidate is not, include `clean_before_add: true` in its config and implement the cleanup in `prepare()` following `mem0_adapter.prepare`.
- The adapter writes nothing to `evaluation/results/` directly — the harness owns that path.

## Failure modes and what to record

When writing the adapter, pre-decide how the next-step (run-bench) skill will classify a failure:

| Symptom at smoke time | Cause | seen_systems.json status |
|---|---|---|
| `ImportError` on candidate package | Rule C: dep not in evaluation-full | `status: failed`, `rejection_reason: "not in evaluation-full dep group"` |
| HTTP 401/403 on `base_url` | candidate requires auth that this env can't provide | `status: failed`, `rejection_reason: "auth required"` |
| `AttributeError: 'CandidateClient' has no attribute 'add'` | TODO 2/3 method names wrong | fix in-place, re-run smoke |
| Candidate returns empty results for all queries on `--smoke` | search API wired incorrectly OR candidate requires background indexing | try `post_add_wait_seconds: 60` in config; if still empty → `status: failed` |
| OOM during smoke with `--smoke-messages 20` | candidate has hidden infra requirement | mark `tier: oversize-infra`, do NOT open PR from this run |
| Candidate works but scores 0 on LoCoMo | adapter is wired but output format mismatches | leave adapter, open PR with `[zero-score]` tag so humans can debug |

## What NOT to do in this skill

- Do NOT run smoke tests. That is the next skill.
- Do NOT open a PR. The routine's main prompt opens the PR at the end.
- Do NOT git commit. The routine's main prompt batches commits.
- Do NOT touch `evermemos_adapter.py`, `evermemos/` subdir, or `openclaw/` subdir.
- Do NOT modify existing system YAMLs.
- Do NOT guess the candidate's API method names — read the README and at least one example from the candidate repo. If the README is ambiguous, prefer the lowest-risk guess (`.add()` for ingest, `.search()` for retrieval) but note the uncertainty in a `# TODO(auto-bench):` comment.
- Do NOT add comments explaining the 4-stage pipeline inside the adapter — the base class docstring already covers it. Keep the adapter file tight.
- Do NOT copy `evermemos_adapter.py` as a starting point — it's the privileged re-implementation path and does not inherit from `OnlineAPIAdapter`. Use `evermemos_api_adapter.py` as the reference instead.
- Do NOT mirror `mem0_adapter.py` — it targets Mem0's SaaS endpoint, which violates Rule 1. Rule 1 was already enforced at the discovery step, so any candidate reaching this skill is local; use the local reference.
