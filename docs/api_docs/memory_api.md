# Memory API Documentation

[Home](../../README.md) > [Docs](../README.md) > [API Docs](.) > Memory API

## Overview

The Memory API provides RESTful endpoints for storing, retrieving, searching, and managing conversational memories.

**Base URL:** `http://localhost:8001/api/v1/memories`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/memories` | Store a single message |
| GET | `/memories` | Fetch memories by type |
| GET | `/memories/search` | Search memories |
| GET | `/memories/conversation-meta` | Get conversation metadata |
| POST | `/memories/conversation-meta` | Save conversation metadata |
| PATCH | `/memories/conversation-meta` | Partial update metadata |
| DELETE | `/memories` | Soft delete memories |

---

## POST `/memories` - Store Message

Store a single message into memory.

### Request

```json
{
  "message_id": "msg_001",
  "create_time": "2025-01-15T10:00:00+00:00",
  "sender": "user_001",
  "content": "Let's discuss the technical solution for the new feature today",
  "group_id": "group_123",
  "group_name": "Project Discussion Group",
  "sender_name": "John",
  "role": "user",
  "refer_list": ["msg_000"]
}
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message_id` | string | Yes | Unique message identifier |
| `create_time` | string | Yes | ISO 8601 timestamp with timezone |
| `sender` | string | Yes | Sender user ID |
| `content` | string | Yes | Message content |
| `group_id` | string | No | Group identifier |
| `group_name` | string | No | Group display name |
| `sender_name` | string | No | Sender display name (defaults to `sender`) |
| `role` | string | No | `user` (human) or `assistant` (AI) |
| `refer_list` | array | No | Referenced message IDs |

### Group ID Behavior

When `group_id` and `group_name` are not provided (null), the API automatically creates a default group based on the `sender` field. This enables simpler use cases where correlated memories between multiple senders are not needed.

**When to omit `group_id`:**
- **Knowledge base ingestion** - Single-source content where sender correlation is not needed
- **Persona/profile building** - Building memories for a single user without multi-party context
- **Simple chatbot interactions** - 1:1 conversations where grouping is not required

**When to provide `group_id`:**
- **Multi-user conversations** - Group chats where multiple participants interact
- **User + AI assistant** - Conversations between a user and AI where context correlation matters
- **Project/topic-based organization** - When you want to query memories by logical groupings

Providing a `group_id` enables better episodic memory extraction by giving the system context about related messages across multiple senders. See the [Group Chat Guide](../advanced/GROUP_CHAT_GUIDE.md) for detailed guidance.

### Example

```bash
curl -X POST "http://localhost:8001/api/v1/memories" \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg_001",
    "create_time": "2025-01-15T10:00:00+00:00",
    "sender": "user_001",
    "sender_name": "John",
    "role": "user",
    "content": "Let us discuss the technical solution for the new feature today",
    "group_id": "group_123",
    "group_name": "Project Discussion Group",
    "refer_list": []
  }'
```

### Response

**Success (200)** - Memory extracted (boundary triggered):
```json
{
  "status": "ok",
  "message": "Extracted 1 memories",
  "result": {
    "saved_memories": [],
    "count": 1,
    "status_info": "extracted"
  }
}
```

**Success (200)** - Message queued (boundary not triggered):
```json
{
  "status": "ok",
  "message": "Message queued, awaiting boundary detection",
  "result": {
    "saved_memories": [],
    "count": 0,
    "status_info": "accumulated"
  }
}
```

---

## GET `/memories` - Fetch Memories

Retrieve memories by type with optional filters.

### Request Parameters (Query String)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `user_id` | string | No* | - | User ID |
| `group_id` | string | No* | - | Group ID |
| `memory_type` | string | No | `episodic_memory` | Memory type |
| `limit` | integer | No | 40 | Max results (max: 500) |
| `offset` | integer | No | 0 | Pagination offset |
| `start_time` | string | No | - | Filter start time (ISO 8601) |
| `end_time` | string | No | - | Filter end time (ISO 8601) |
| `version_range` | array | No | - | Version range `[start, end]` |

*At least one of `user_id` or `group_id` must be provided (cannot both be `__all__`).

### Memory Types

| Type | Description |
|------|-------------|
| `profile` | User profile information |
| `episodic_memory` | Conversation episodes (default) |
| `foresight` | Prospective memory |
| `event_log` | Atomic facts |

### Example

```bash
curl "http://localhost:8001/api/v1/memories?user_id=user_123&memory_type=episodic_memory&limit=20"
```

### Response

```json
{
  "status": "ok",
  "message": "Memory retrieval successful, retrieved 1 memories",
  "result": {
    "memories": [
      {
        "memory_type": "episodic_memory",
        "user_id": "user_123",
        "timestamp": "2024-01-15T10:30:00",
        "content": "User discussed coffee during the project sync",
        "summary": "Project sync coffee note"
      }
    ],
    "total_count": 100,
    "has_more": false,
    "metadata": {
      "source": "fetch_mem_service",
      "user_id": "user_123",
      "memory_type": "fetch"
    }
  }
}
```

---

## GET `/memories/search` - Search Memories

Search memories using keyword, vector, or hybrid retrieval methods.

### Request Body

```json
{
  "query": "coffee preference",
  "user_id": "user_123",
  "group_id": "group_456",
  "retrieve_method": "keyword",
  "memory_types": ["episodic_memory"],
  "top_k": 10,
  "start_time": "2024-01-01T00:00:00",
  "end_time": "2024-12-31T23:59:59",
  "radius": 0.6,
  "include_metadata": true
}
```

### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | No | - | Search query text |
| `user_id` | string | No* | - | User ID |
| `group_id` | string | No* | - | Group ID |
| `retrieve_method` | string | No | `keyword` | Retrieval method |
| `memory_types` | array | No | `[]` (defaults to `episodic_memory`) | Memory types to search |
| `top_k` | integer | No | 40 | Max results (max: 100) |
| `start_time` | string | No | - | Filter start time (ISO 8601) |
| `end_time` | string | No | - | Filter end time (ISO 8601) |
| `radius` | float | No | - | Cosine similarity threshold (0.0-1.0, for vector/hybrid only) |
| `include_metadata` | boolean | No | true | Include metadata in response |
| `current_time` | string | No | - | Current time for filtering foresight events |

*At least one of `user_id` or `group_id` must be provided (cannot both be `__all__`).

**Note:** `profile` memory type is not supported in the search interface.

### Retrieve Methods

| Method | Description |
|--------|-------------|
| `keyword` | BM25 keyword retrieval (default) |
| `vector` | Vector semantic retrieval |
| `hybrid` | Keyword + vector + rerank |
| `rrf` | RRF fusion (keyword + vector + RRF ranking) |
| `agentic` | LLM-guided multi-round intelligent retrieval |

### Example

```bash
curl -X GET "http://localhost:8001/api/v1/memories/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "coffee preference",
    "user_id": "user_123",
    "retrieve_method": "keyword",
    "top_k": 10
  }'
```

### Response

```json
{
  "status": "ok",
  "message": "Memory search successful, retrieved 1 groups",
  "result": {
    "memories": [
      {
        "episodic_memory": [
          {
            "memory_type": "episodic_memory",
            "user_id": "user_123",
            "timestamp": "2024-01-15T10:30:00",
            "summary": "Discussed coffee choices",
            "group_id": "group_456"
          }
        ]
      }
    ],
    "scores": [{"episodic_memory": [0.95]}],
    "importance_scores": [0.85],
    "original_data": [],
    "total_count": 45,
    "has_more": false,
    "query_metadata": {
      "source": "episodic_memory_es_repository",
      "user_id": "user_123",
      "memory_type": "retrieve"
    },
    "metadata": {
      "source": "episodic_memory_es_repository",
      "user_id": "user_123",
      "memory_type": "retrieve"
    },
    "pending_messages": []
  }
}
```

### Response Fields

| Field | Description |
|-------|-------------|
| `memories` | List of memory groups, organized by memory type |
| `scores` | Relevance scores for each memory |
| `importance_scores` | Group importance scores for sorting |
| `original_data` | Original data associated with memories |
| `total_count` | Total number of memories found |
| `has_more` | Whether more results are available |
| `query_metadata` | Metadata about the query execution |
| `metadata` | Additional response metadata |
| `pending_messages` | Messages waiting for memory extraction |

---

## GET `/memories/conversation-meta` - Get Metadata

Retrieve conversation metadata by group_id with fallback to default config.

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `group_id` | string | No | Group ID (omit for default config) |

### Example

```bash
curl "http://localhost:8001/api/v1/memories/conversation-meta?group_id=group_123"
```

### Response

```json
{
  "status": "ok",
  "message": "Conversation metadata retrieved successfully",
  "result": {
    "id": "...",
    "group_id": "group_123",
    "scene": "group_chat",
    "name": "Engineering Team",
    "user_details": {...},
    "is_default": false
  }
}
```

---

## POST `/memories/conversation-meta` - Save Metadata

Save or update conversation metadata (upsert behavior).

### Request Body

```json
{
  "version": "1.0.0",
  "scene": "group_chat",
  "scene_desc": {
    "description": "Project discussion group chat",
    "type": "project_discussion"
  },
  "name": "Engineering Team",
  "description": "Backend team discussions",
  "group_id": "group_123",
  "created_at": "2025-01-15T10:00:00Z",
  "default_timezone": "America/New_York",
  "user_details": {
    "alice": {
      "full_name": "Alice Smith",
      "role": "user",
      "custom_role": "Tech Lead"
    }
  },
  "tags": ["engineering", "backend"]
}
```

### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Metadata version |
| `scene` | string | Yes | Scene identifier: `assistant` or `group_chat` |
| `scene_desc` | object | Yes | Scene description object |
| `name` | string | Yes | Conversation name |
| `description` | string | No | Conversation description |
| `group_id` | string | No | Group identifier (omit for default config) |
| `created_at` | string | Yes | Conversation creation time (ISO 8601 format) |
| `default_timezone` | string | No | Default timezone (defaults to system timezone) |
| `user_details` | object | No | Participant details, key is user ID |
| `tags` | array | No | Tag list |

### User Details Fields

| Field | Type | Description |
|-------|------|-------------|
| `full_name` | string | Display name |
| `role` | string | `user` or `assistant` |
| `custom_role` | string | Job title/position |
| `extra` | object | Additional metadata |

---

## PATCH `/memories/conversation-meta` - Update Metadata

Partially update conversation metadata.

### Request Body

```json
{
  "group_id": "group_123",
  "name": "Updated Team Name",
  "tags": ["engineering", "python"]
}
```

### Updatable Fields

| Field | Description |
|-------|-------------|
| `name` | Conversation name |
| `description` | Conversation description |
| `scene_desc` | Scene description |
| `tags` | Tag list |
| `user_details` | User details (replaces entire object) |
| `default_timezone` | Default timezone |

### Response

```json
{
  "status": "ok",
  "message": "Conversation metadata updated successfully, 2 fields updated",
  "result": {
    "id": "...",
    "group_id": "group_123",
    "scene": "group_chat",
    "name": "Updated Team Name",
    "updated_fields": ["name", "tags"],
    "updated_at": "2025-01-15T12:00:00Z"
  }
}
```

---

## DELETE `/memories` - Delete Memories

Soft delete memories based on filter criteria (AND logic).

### Request Body

```json
{
  "event_id": "evt_001",
  "user_id": "user_123",
  "group_id": "group_456"
}
```

### Request Fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `event_id` | string | No | `__all__` | Filter by event ID |
| `user_id` | string | No | `__all__` | Filter by user ID |
| `group_id` | string | No | `__all__` | Filter by group ID |

At least one filter must be provided (not all `__all__`).

### Example

```bash
# Delete all memories for a user in a group
curl -X DELETE "http://localhost:8001/api/v1/memories" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "group_id": "group_456"}'
```

### Response

```json
{
  "status": "ok",
  "message": "Successfully deleted 10 memories",
  "result": {
    "filters": ["user_id", "group_id"],
    "count": 10
  }
}
```

---

## Batch Processing with run_memorize.py

For batch processing GroupChatFormat JSON files:

```bash
# Process a group chat file
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --scene group_chat \
  --api-url http://localhost:8001/api/v1/memories

# Validate format only
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --scene group_chat \
  --validate-only
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--input` | Yes | Path to GroupChatFormat JSON file |
| `--scene` | Yes | `group_chat` or `assistant` |
| `--api-url` | Yes* | Memory API endpoint |
| `--validate-only` | No | Only validate format, skip processing |

*Required unless using `--validate-only`.

---

## Error Responses

All error responses follow this format:

```json
{
  "status": "failed",
  "code": "ERROR_CODE",
  "message": "Human-readable error message",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v1/memories"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_PARAMETER` | 400 | Invalid or missing request parameters |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource not found |
| `SYSTEM_ERROR` | 500 | Internal server error |

---

## See Also

- [Group Chat Guide](../advanced/GROUP_CHAT_GUIDE.md) - Multi-participant conversations
- [Metadata Control Guide](../advanced/METADATA_CONTROL.md) - Conversation metadata management
- [GroupChatFormat Specification](../../data_format/group_chat/group_chat_format.md) - Data format reference
