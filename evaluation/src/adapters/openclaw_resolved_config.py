"""
OpenClaw-schema config builder for the benchmark adapter.

Produces the dict shape that OpenClaw's CLI reads via OPENCLAW_CONFIG_PATH.

We emit only the fields we actually override; everything else is left out so
OpenClaw's own defaults from src/agents/memory-search.ts (tokenizer,
chunking, cache, sync debounce, hybrid weights, mmr/temporal-decay toggles,
etc.) apply. Native defaults we explicitly mirror here have citations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


# Citations from /Data3/shutong.shan/openclaw/repo for reviewers:
#   memorySearch.maxResults default ............ memory-search.ts:103 -> 6
#   memorySearch.minScore default .............. memory-search.ts:104 -> 0.35
#   memorySearch.sync.onSearch default ......... memory-search.ts:234 -> true
#   memorySearch.sync.onSessionStart default ... memory-search.ts:233 -> true
#   memorySearch.sync.watch default ............ memory-search.ts:235 -> true
#   memorySearch.store.vector.enabled default .. memory-search.ts:215 -> true
#   hybrid.vectorWeight / textWeight ........... memory-search.ts:106/107
#   hybrid.candidateMultiplier ................. memory-search.ts:108 -> 4
#   chunking.tokens / overlap .................. memory-search.ts:98/99
#   cache.enabled .............................. memory-search.ts:113 -> true
#   compaction.memoryFlush.softThresholdTokens . flush-plan.ts:10    -> 4000
#   compaction.memoryFlush.forceFlushTranscriptBytes ... flush-plan.ts:11 -> 2MB
#   compaction.reserveTokensFloor .............. pi-settings.ts:4    -> 20000
#   memory.backend ............................. backend-config.ts:79 -> "builtin"


def build_openclaw_resolved_config(
    *,
    workspace_dir: str,
    native_store_dir: str,
    backend_mode: str,
    flush_mode: str,
    embedding: Optional[dict] = None,
) -> dict:
    """Return the dict that OpenClaw CLI expects at OPENCLAW_CONFIG_PATH.

    The benchmark deliberately deviates from OpenClaw's upstream defaults on
    exactly one switch: ``compaction.memoryFlush.enabled`` stays false
    because our adapter drives flush at ingest time and we do not want
    OpenClaw to double-flush during search.

    backend_mode:
        ``fts_only``: provider=auto + vector disabled (no embedding)
        ``vector``:   provider=sophnet + vector enabled (embedding only)
        ``hybrid``:   provider=sophnet + vector enabled (BM25 + embeddings,
                      OpenClaw's production retrieval path)
    """
    sqlite_path = str(Path(native_store_dir) / "memory" / "default.sqlite")

    # Let OpenClaw apply its own defaults to tokenizer / sync flags /
    # chunking / cache / hybrid weights unless we have a specific reason
    # to pin a value. The only pin is vector.enabled because it is not a
    # scalar default - it is a switch that depends on our backend_mode.
    memory_search: dict[str, Any] = {
        "store": {
            "path": sqlite_path,
            "vector": {"enabled": backend_mode != "fts_only"},
        },
        "sources": ["memory"],
    }

    if backend_mode == "fts_only":
        memory_search["provider"] = "auto"
    else:
        memory_search["provider"] = (embedding or {}).get("provider", "sophnet")
        memory_search["model"] = (embedding or {}).get("model", "text-embeddings")
        memory_search["outputDimensionality"] = int(
            (embedding or {}).get("output_dimensionality", 1024)
        )
        remote = {
            "baseUrl": (embedding or {}).get("base_url", ""),
            "easyllmId": (embedding or {}).get("easyllm_id", ""),
            "apiKey": (embedding or {}).get("api_key", ""),
        }
        memory_search["remote"] = remote

    # Only deliberate deviation from OpenClaw native: keep the in-search
    # flush off because we drive flush ourselves at ingest. Everything
    # else under compaction.* (softThresholdTokens, reserveTokensFloor,
    # forceFlushTranscriptBytes) is left implicit so OpenClaw applies its
    # own defaults.
    memory_flush_enabled = False

    return {
        "memory": {"backend": "builtin"},
        "agents": {
            "defaults": {
                "workspace": workspace_dir,
                "userTimezone": "UTC",
                "memorySearch": memory_search,
                "compaction": {
                    "memoryFlush": {"enabled": memory_flush_enabled},
                },
            }
        },
    }
