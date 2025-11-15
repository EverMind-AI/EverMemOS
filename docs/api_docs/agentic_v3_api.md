# Agentic Layer V3 API Documentation

## Overview

Agentic Layer V3 exposes FastAPI routes under `/api/v3/agentic` for ingesting chat messages and retrieving memories. All endpoints accept `application/json` and return a body shaped as:

```
{
  "status": "ok",
  "message": "<human readable summary>",
  "result": { ... }
}
```

`status` becomes `failed` when validation or server errors occur.

## Endpoint Summary

| Endpoint | Description |
|----------|-------------|
| `POST /api/v3/agentic/memorize` | Persist a single chat event as memcells |
| `POST /api/v3/agentic/retrieve_lightweight` | Low-latency hybrid retrieval (Embedding + BM25 + RRF) |
| `POST /api/v3/agentic/retrieve_agentic` | LLM-guided multi-round retrieval with rerank |

---

## POST `/api/v3/agentic/memorize`

Stores one message without any pre-conversion. When `group_id` is provided, the raw payload is also appended to `chat_history:{group_id}` in Redis (left push, TTL 24h) for incremental boundary detection.

### Request Body

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `group_id` | string | No | Enables Redis history accumulation |
| `group_name` | string | No | Informational name |
| `message_id` | string | Yes | Must be unique per message |
| `create_time` | string | Yes | ISO 8601 timestamp with optional timezone |
| `sender` | string | Yes | Sender user ID |
| `sender_name` | string | No | Defaults to `sender` |
| `content` | string | Yes | Raw message content |
| `refer_list` | array\<string> | No | Referenced message IDs |

### Success Response

```
{
  "status": "ok",
  "message": "Extracted 1 memories",
  "result": {
    "saved_memories": [
      {
        "memory_type": "episode_summary",
        "user_id": "user_001",
        "group_id": "group_123",
        "timestamp": "2025-01-15T10:00:00",
        "content": "User discussed the new feature plan"
      }
    ],
    "count": 1,
    "status_info": "extracted"
  }
}
```

`status_info` becomes `accumulated` when boundary detection delays extraction; in that case `saved_memories` is empty but the raw message has been queued safely.

### Errors

- `400 Bad Request`: missing or malformed fields. FastAPI responds with `{"detail": "<reason>"}`.
- `500 Internal Server Error`: unexpected failure while converting, storing, or calling downstream services.

---

## POST `/api/v3/agentic/retrieve_lightweight`

Runs embedding retrieval and BM25 in parallel, fuses them with Reciprocal Rank Fusion (RRF), and returns the merged list.

### Request Fields

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `query` | string | Yes* | — | Optional only when `data_source="profile"` |
| `user_id` | string | No | `null` | Filter for personal memories |
| `group_id` | string | No | `null` | Filter for group conversations |
| `time_range_days` | integer | No | `365` | Sliding window over timestamps |
| `top_k` | integer | No | `20` | Maximum number of rows returned |
| `retrieval_mode` | string | No | `rrf` | `rrf`, `embedding`, or `bm25` |
| `data_source` | string | No | `episode` | `episode`, `event_log`, `semantic_memory`, or `profile`; `memcell` aliases to `episode` |
| `memory_scope` | string | No | `all` | `all`, `personal`, or `group` |
| `current_time` | string | No | `null` | `YYYY-MM-DD`, used for semantic-memory freshness |
| `radius` | float | No | `null` | Cosine similarity threshold in `[-1, 1]` |

\* For `data_source="profile"` both `user_id` and `group_id` must be provided even though `query` can be omitted.

### Success Response

```
{
  "status": "ok",
  "message": "检索成功，找到 10 条记忆",
  "result": {
    "memories": [...],
    "count": 10,
    "metadata": {
      "retrieval_mode": "lightweight",
      "emb_count": 15,
      "bm25_count": 12,
      "final_count": 10,
      "total_latency_ms": 123.45
    }
  }
}
```

### Errors

- `400 Bad Request`: invalid `current_time`, missing `query`, missing profile identifiers, etc.
- `500 Internal Server Error`: downstream failure while executing retrieval.

---

## POST `/api/v3/agentic/retrieve_agentic`

Executes an agentic workflow: initial hybrid search, rerank, LLM sufficiency check, optional query refinement, a second retrieval round, and final rerank.

### Request Fields

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `query` | string | Yes | — | Natural-language prompt |
| `user_id` | string | No | `null` | Filter for personal memories |
| `group_id` | string | No | `null` | Filter for group memories |
| `time_range_days` | integer | No | `365` | Temporal window |
| `top_k` | integer | No | `20` | Final number of memories |
| `llm_config.api_key` | string | Yes* | — | OpenAI-compatible API key |
| `llm_config.base_url` | string | No | `https://openrouter.ai/api/v1` | Override for OpenAI-compatible endpoint |
| `llm_config.model` | string | No | `qwen/qwen3-235b-a22b-2507` | Model identifier consumed by `LLMProvider` |

\* The API key may also come from `OPENROUTER_API_KEY` or `OPENAI_API_KEY`. Without any key the controller raises `400`.

### Success Response

```
{
  "status": "ok",
  "message": "Agentic 检索成功，找到 15 条记忆",
  "result": {
    "memories": [...],
    "count": 15,
    "metadata": {
      "retrieval_mode": "agentic",
      "is_multi_round": true,
      "round1_count": 20,
      "is_sufficient": false,
      "reasoning": "需要更多关于饮食偏好的具体信息",
      "refined_queries": [
        "用户最喜欢的菜系？",
        "用户不喜欢吃什么？"
      ],
      "round2_count": 40,
      "final_count": 15,
      "total_latency_ms": 2345.67
    }
  }
}
```

### Errors

- `400 Bad Request`: missing `query`, missing API key, or malformed parameters.
- `500 Internal Server Error`: failures when communicating with the LLM provider or memory subsystem.

---

## Error Payloads

When a request fails, the service raises `HTTPException`. Downstream exception handlers format the payload as:

```
{
  "status": "failed",
  "code": "<ErrorCode>",
  "message": "<readable error>",
  "timestamp": "<ISO 8601>",
  "path": "<request path>"
}
```

Implement retries on `5xx` responses if the client can tolerate duplicate work.

