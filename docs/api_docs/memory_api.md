# Memory API Documentation

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Interface Specification](#interface-specification)
  - [POST `/api/v1/memories` - Store Single Message Memory](#post-apiv1memories)
- [Use Cases](#use-cases)
  - [1. Real-time Message Stream Processing](#1-real-time-message-stream-processing)
  - [2. Chatbot Integration](#2-chatbot-integration)
  - [3. Message Queue Consumption](#3-message-queue-consumption)
- [Usage Examples](#usage-examples)
  - [Using curl](#using-curl)
  - [Using Python Code](#using-python-code)
  - [Using run_memorize.py Script](#using-run_memorizepy-script)
- [FAQ](#faq)
- [Architecture](#architecture)
  - [Data Flow](#data-flow)
  - [Core Components](#core-components)
- [Memory Query Interfaces](#memory-query-interfaces)
  - [GET `/api/v1/memories` - Fetch User Memories](#get-apiv1memories)
  - [GET `/api/v1/memories/search` - Search Memories](#get-apiv1memoriessearch)
- [Conversation Metadata Management](#conversation-metadata-management)
  - [POST `/api/v1/memories/conversation-meta` - Save Conversation Metadata](#post-apiv1memoriesconversation-meta)
  - [PATCH `/api/v1/memories/conversation-meta` - Partial Update Conversation Metadata](#patch-apiv1memoriesconversation-meta)
- [Related Documentation](#related-documentation)

---

## Overview

The Memory API provides specialized interfaces for processing group chat memories, using a simple and direct message format without any preprocessing or format conversion.

## Key Features

- ✅ **Simple and Direct**: Uses the simplest single message format, no complex data structures required
- ✅ **No Conversion Needed**: No format conversion or adaptation required
- ✅ **Sequential Processing**: Real-time processing of each message, suitable for message stream scenarios
- ✅ **Centralized Adaptation**: All format conversion logic centralized in `group_chat_converter.py`, maintaining single responsibility
- ✅ **Detailed Error Messages**: Provides clear error prompts and data statistics

## Interface Specification

### POST `/api/v1/memories`

Store a single group chat message memory

#### Request Format

**Content-Type**: `application/json`

**Request Body**: Simple direct single message format (no pre-conversion needed)

```json
{
  "group_id": "group_123",
  "group_name": "Project Discussion Group",
  "message_id": "msg_001",
  "create_time": "2025-01-15T02:00:00Z",
  "sender": "user_001",
  "sender_name": "Zhang San",
  "content": "Let's discuss the technical approach for the new feature today",
  "refer_list": ["msg_000"]
}
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| group_id | string | No | Group ID |
| group_name | string | No | Group name |
| message_id | string | Yes | Unique message identifier |
| create_time | string | Yes | Message creation time (ISO 8601 format) |
| sender | string | Yes | Sender user ID |
| sender_name | string | No | Sender name (defaults to sender) |
| content | string | Yes | Message content |
| refer_list | array | No | List of referenced message IDs |

#### Response Format

**Success Response (200 OK)**

The response has two forms depending on the memory extraction status:

**1. Extracted** - When the message triggers boundary detection and successfully extracts memories:

```json
{
  "status": "ok",
  "message": "Extracted 1 memories",
  "result": {
    "saved_memories": [
      {
        "memory_type": "episode_memory",
        "user_id": "user_001",
        "group_id": "group_123",
        "timestamp": "2025-01-15T10:00:00",
        "content": "User discussed technical approach for the new feature"
      }
    ],
    "count": 1,
    "status_info": "extracted"
  }
}
```

**2. Accumulated** - When the message is stored but does not trigger boundary detection:

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

**Field Descriptions**:
- `status_info`: Processing status, `extracted` means memories were extracted, `accumulated` means message is queued awaiting boundary detection
- `saved_memories`: List of saved memories, empty array when `status_info` is `accumulated`
- `count`: Number of saved memories

**Error Response (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "Data format error: missing required field message_id",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v1/memories"
}
```

**Error Response (500 Internal Server Error)**

```json
{
  "status": "failed",
  "code": "SYSTEM_ERROR",
  "message": "Failed to store memory, please try again later",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v1/memories"
}
```

---

## Use Cases

### 1. Real-time Message Stream Processing

Suitable for processing real-time message streams from chat applications, storing each message as it arrives.

**Example**:
```json
{
  "group_id": "group_123",
  "group_name": "Project Discussion Group",
  "message_id": "msg_001",
  "create_time": "2025-01-15T02:00:00Z",
  "sender": "user_001",
  "sender_name": "Zhang San",
  "content": "Let's discuss the technical approach for the new feature today",
  "refer_list": []
}
```

### 2. Chatbot Integration

After a chatbot receives a user message, it can directly call the Memory API to store the memory.

**Example**:
```json
{
  "group_id": "bot_conversation_123",
  "group_name": "Conversation with AI Assistant",
  "message_id": "bot_msg_001",
  "create_time": "2025-01-15T02:05:00Z",
  "sender": "user_456",
  "sender_name": "Li Si",
  "content": "Help me summarize today's meeting content",
  "refer_list": []
}
```

### 3. Message Queue Consumption

When consuming messages from a message queue (such as Kafka), you can call the Memory API for each message.

**Kafka Consumer Example**:
```python
from kafka import KafkaConsumer
import httpx
import asyncio

async def process_message(message):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v1/memories",
            json={
                "group_id": message["group_id"],
                "group_name": message["group_name"],
                "message_id": message["message_id"],
                "create_time": message["create_time"],
                "sender": message["sender"],
                "sender_name": message["sender_name"],
                "content": message["content"],
                "refer_list": message.get("refer_list", [])
            }
        )
        return response.json()

# Kafka consumer
consumer = KafkaConsumer('chat_messages')
for msg in consumer:
    asyncio.run(process_message(msg.value))
```

---

## Usage Examples

### Using curl

```bash
curl -X POST http://localhost:1995/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "group_name": "Project Discussion Group",
    "message_id": "msg_001",
    "create_time": "2025-01-15T02:00:00Z",
    "sender": "user_001",
    "sender_name": "Zhang San",
    "content": "Let'\''s discuss the technical approach for the new feature today",
    "refer_list": []
  }'
```

### Using Python Code

```python
import httpx
import asyncio

async def call_memory_api():
    # Simple direct single message format
    message_data = {
        "group_id": "group_123",
        "group_name": "Project Discussion Group",
        "message_id": "msg_001",
        "create_time": "2025-01-15T02:00:00Z",
        "sender": "user_001",
        "sender_name": "Zhang San",
        "content": "Let's discuss the technical approach for the new feature today",
        "refer_list": []
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v1/memories",
            json=message_data
        )
        result = response.json()
        print(f"Saved {result['result']['count']} memories")

asyncio.run(call_memory_api())
```

### Using run_memorize.py Script

For JSON files in GroupChatFormat, you can use the `run_memorize.py` script for batch processing:

```bash
# Store memory
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories

# Validate format only
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --validate-only
```

---

## FAQ

### 1. How to handle messages with references?

Use the `refer_list` field to specify the list of referenced message IDs:

```json
{
  "message_id": "msg_002",
  "content": "I agree with your approach",
  "refer_list": ["msg_001"]
}
```

### 3. Are group_id and group_name required?

Not required, but **strongly recommended**:
- `group_id` is used to identify the group for easier retrieval
- `group_name` is used for display and understanding, improving readability

### 4. How to handle private chat messages?

Private chat messages can omit `group_id`, or use a special private chat ID:

```json
{
  "group_id": "private_user001_user002",
  "group_name": "Private chat with Zhang San",
  "message_id": "private_msg_001",
  "create_time": "2025-01-15T02:00:00Z",
  "sender": "user_001",
  "sender_name": "Zhang San",
  "content": "Hi, how are you doing?",
  "refer_list": []
}
```

### 5. How to handle message timestamps?

`create_time` must use ISO 8601 format, timezone support:

```json
{
  "create_time": "2025-01-15T02:00:00Z"  // with timezone
}
```

Or without timezone (defaults to UTC):

```json
{
  "create_time": "2025-01-15T10:00:00"  // UTC
}
```

### 6. How to batch process historical messages?

Use the `run_memorize.py` script:

1. Prepare a JSON file in GroupChatFormat
2. Run the script, which will automatically call the Memory API for each message

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories
```

### 7. Are there rate limits for API calls?

Currently no hard limits, but we recommend:
- Real-time scenarios: No more than 100 requests per second
- Batch import: Suggest 0.1 second interval between messages

### 8. How to handle errors?

The interface returns detailed error messages:

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "Missing required field: message_id"
}
```

We recommend implementing retry mechanism on the client side, with up to 3 retries for 5xx errors.

---

## Architecture

### Data Flow

```
Client
  ↓
  │ Simple direct single message format
  ↓
Memory Controller (memory_controller.py)
  ↓
  │ Call group_chat_converter.py
  ↓
Format Conversion (convert_simple_message_to_memorize_input)
  ↓
  │ Internal format
  ↓
Memory Manager (memory_manager.py)
  ↓
  │ Memory storage
  ↓
Database / Vector Database
```

### Core Components

1. **Memory Controller** (`memory_controller.py`)
   - Receives simple direct single messages
   - Calls converter for format conversion
   - Calls memory_manager to store memories

2. **Group Chat Converter** (`group_chat_converter.py`)
   - Centralized adaptation layer
   - Responsible for all format conversion logic
   - Maintains single responsibility

3. **Memory Manager** (`memory_manager.py`)
   - Memory extraction and storage
   - Vectorization
   - Persistence

---

## Memory Query Interfaces

### GET `/api/v1/memories`

Retrieve user's core memory data through KV method.

#### Features

- Directly retrieve stored core memories based on user ID
- Support multiple memory types: base memory, profile, preferences, etc.
- Support pagination and sorting
- Suitable for scenarios requiring quick access to user's fixed memory collection

#### Request Parameters (Query Parameters)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| user_id | string | Yes | - | User ID |
| memory_type | string | No | "multiple" | Memory type, options: `base_memory`, `profile`, `preference`, `episode_memory`, `multiple` |
| limit | integer | No | 40 | Maximum number of memories to return |
| offset | integer | No | 0 | Pagination offset |
| sort_by | string | No | - | Sort field |
| sort_order | string | No | "desc" | Sort direction, options: `asc`, `desc` |

**Memory Type Descriptions**:
- `base_memory`: Base memory, user's basic information and common data
- `profile`: User profile, containing user's characteristics and attributes
- `preference`: User preferences, containing user's likes and settings
- `episode_memory`: Episodic memory summary
- `multiple`: Multiple types (default), includes base_memory, profile, preference

#### Response Format

**Success Response (200 OK)**

```json
{
  "status": "ok",
  "message": "记忆获取成功，共获取 15 条记忆",
  "result": {
    "memories": [
      {
        "memory_type": "base_memory",
        "user_id": "user_123",
        "timestamp": "2024-01-15T10:30:00",
        "content": "User likes coffee",
        "summary": "Coffee preference"
      },
      {
        "memory_type": "profile",
        "user_id": "user_123",
        "timestamp": "2024-01-14T09:20:00",
        "content": "User is a software engineer"
      }
    ],
    "total_count": 15,
    "has_more": false,
    "metadata": {
      "source": "fetch_mem_service",
      "user_id": "user_123",
      "memory_type": "fetch"
    }
  }
}
```

#### Usage Examples

**Using curl**:

```bash
curl -X GET "http://localhost:1995/api/v1/memories?user_id=user_123&memory_type=multiple&limit=20" \
  -H "Content-Type: application/json"
```

**Using Python**:

```python
import httpx
import asyncio

async def fetch_memories():
    params = {
        "user_id": "user_123",
        "memory_type": "multiple",
        "limit": 20,
        "offset": 0
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:1995/api/v1/memories",
            params=params
        )
        result = response.json()
        print(f"Fetched {len(result['result']['memories'])} memories")

asyncio.run(fetch_memories())
```

---

### GET `/api/v1/memories/search`

Retrieve relevant memories based on query text using keyword, vector, or hybrid methods.

#### Features

- Find most relevant memories based on query text
- Support three retrieval methods: keyword (BM25), vector similarity, and hybrid
- Support time range filtering
- Results organized by groups with relevance scores
- Suitable for scenarios requiring exact matching or semantic retrieval

#### Request Format

**Content-Type**: `application/json`

**Request Body**:

```json
{
  "user_id": "user_123",
  "query": "coffee preference",
  "retrieve_method": "keyword",
  "top_k": 10,
  "start_time": "2024-01-01T00:00:00",
  "end_time": "2024-12-31T23:59:59",
  "memory_types": ["episode_memory"],
  "filters": {
    "group_id": "group_456"
  }
}
```

**Field Descriptions**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| user_id | string | Yes | - | User ID |
| query | string | No | - | Query text |
| retrieve_method | string | No | "keyword" | Retrieval method, options: `keyword`, `vector`, `hybrid` |
| top_k | integer | No | 40 | Maximum number of results to return |
| start_time | string | No | - | Time range start (ISO 8601 format) |
| end_time | string | No | - | Time range end (ISO 8601 format) |
| memory_types | array | No | [] | List of memory types to retrieve |
| filters | object | No | {} | Additional filter conditions |
| radius | float | No | 0.6 | Similarity threshold for vector retrieval (only for vector and hybrid methods) |

**Retrieval Method Descriptions**:
- `keyword`: BM25 keyword-based retrieval, suitable for exact matching, fast (default)
- `vector`: Semantic vector similarity retrieval, suitable for fuzzy queries and semantic similarity
- `hybrid`: Hybrid retrieval strategy combining keyword and vector advantages (recommended)

#### Response Format

**Success Response (200 OK)**

```json
{
  "status": "ok",
  "message": "记忆检索成功，共检索到 2 个群组",
  "result": {
    "memories": [
      {
        "group_456": [
          {
            "memory_type": "episode_memory",
            "user_id": "user_123",
            "timestamp": "2024-01-15T10:30:00",
            "summary": "Discussed coffee preferences",
            "group_id": "group_456"
          }
        ]
      }
    ],
    "scores": [
      {
        "group_456": [0.95]
      }
    ],
    "importance_scores": [0.85],
    "original_data": [],
    "total_count": 2,
    "has_more": false,
    "query_metadata": {
      "source": "episodic_memory_es_repository",
      "user_id": "user_123",
      "memory_type": "retrieve"
    }
  }
}
```

**Return Field Descriptions**:
- `memories`: Memory list organized by groups
- `scores`: Relevance score for each memory
- `importance_scores`: Group importance scores used for sorting
- `total_count`: Total number of memories
- `has_more`: Whether there are more results

#### Usage Examples

**Using curl**:

```bash
curl -X GET http://localhost:1995/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "query": "coffee preference",
    "retrieve_method": "keyword",
    "top_k": 10
  }'
```

**Using Python**:

```python
import httpx
import asyncio

async def search_memories():
    search_data = {
        "user_id": "user_123",
        "query": "coffee preference",
        "retrieve_method": "hybrid",
        "top_k": 10,
        "memory_types": ["episode_memory"]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:1995/api/v1/memories/search",
            json=search_data
        )
        result = response.json()
        print(f"Found {result['result']['total_count']} memories")

asyncio.run(search_memories())
```

---

## Conversation Metadata Management

### POST `/api/v1/memories/conversation-meta`

Save conversation metadata, including scene, participants, tags, etc.

#### Request Format

**Content-Type**: `application/json`

**Request Body**:

```json
{
  "version": "1.0",
  "scene": "group_chat",
  "scene_desc": "Project team discussion",
  "name": "Project Discussion Group",
  "description": "Technical discussion for new feature development",
  "group_id": "group_123",
  "created_at": "2025-01-15T02:00:00Z",
  "default_timezone": "UTC",
  "user_details": {
    "user_001": {
      "full_name": "Zhang San",
      "role": "developer",
      "extra": {"department": "Engineering"}
    },
    "user_002": {
      "full_name": "Li Si",
      "role": "designer",
      "extra": {"department": "Design"}
    }
  },
  "tags": ["work", "technical"]
}
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| version | string | Yes | Metadata version |
| scene | string | Yes | Scene identifier (e.g., "group_chat") |
| scene_desc | string | Yes | Scene description |
| name | string | Yes | Conversation name |
| description | string | Yes | Conversation description |
| group_id | string | Yes | Unique group identifier |
| created_at | string | Yes | Conversation creation time (ISO 8601 format) |
| default_timezone | string | Yes | Default timezone |
| user_details | object | Yes | Participant details |
| tags | array | No | Tag list |

#### Response Format

**Success Response (200 OK)**

```json
{
  "status": "ok",
  "message": "Conversation metadata saved successfully",
  "result": {
    "id": "507f1f77bcf86cd799439011",
    "group_id": "group_123",
    "scene": "group_chat",
    "name": "Project Discussion Group",
    "version": "1.0",
    "created_at": "2025-01-15T02:00:00Z",
    "updated_at": "2025-01-15T02:00:00Z"
  }
}
```

**Note**: This interface uses upsert behavior. If `group_id` already exists, it will update the entire record.

---

### PATCH `/api/v1/memories/conversation-meta`

Partially update conversation metadata, only updating the provided fields.

#### Request Format

**Content-Type**: `application/json`

**Request Body** (only provide fields to update):

```json
{
  "group_id": "group_123",
  "name": "New Conversation Name",
  "tags": ["tag1", "tag2"]
}
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| group_id | string | Yes | Group ID to update |
| name | string | No | New conversation name |
| description | string | No | New description |
| scene_desc | string | No | New scene description |
| tags | array | No | New tag list |
| user_details | object | No | New user details (completely replaces existing user_details) |
| default_timezone | string | No | New default timezone |

**Updatable Fields**:
- `name`: Conversation name
- `description`: Conversation description
- `scene_desc`: Scene description
- `tags`: Tag list
- `user_details`: User details (completely replaces existing user_details)
- `default_timezone`: Default timezone

**Immutable Fields** (cannot be modified via PATCH):
- `version`: Metadata version
- `scene`: Scene identifier
- `group_id`: Group ID
- `conversation_created_at`: Conversation creation time

#### Response Format

**Success Response (200 OK)**

```json
{
  "status": "ok",
  "message": "Conversation metadata updated successfully, 2 fields updated",
  "result": {
    "id": "507f1f77bcf86cd799439011",
    "group_id": "group_123",
    "scene": "group_chat",
    "name": "New Conversation Name",
    "updated_fields": ["name", "tags"],
    "updated_at": "2025-01-15T02:30:00Z"
  }
}
```

**Error Response (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "Missing required field group_id",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v1/memories/conversation-meta"
}
```

**Error Response (404 Not Found)**

```json
{
  "status": "failed",
  "code": "RESOURCE_NOT_FOUND",
  "message": "Conversation metadata not found: group_123",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v1/memories/conversation-meta"
}
```

#### Usage Example

**Using curl**:

```bash
# Partially update conversation metadata
curl -X PATCH http://localhost:1995/api/v1/memories/conversation-meta \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "name": "New Conversation Name",
    "tags": ["updated", "tags"]
  }'
```

**Using Python**:

```python
import httpx
import asyncio

async def patch_conversation_meta():
    update_data = {
        "group_id": "group_123",
        "name": "New Conversation Name",
        "tags": ["updated", "tags"]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.patch(
            "http://localhost:1995/api/v1/memories/conversation-meta",
            json=update_data
        )
        result = response.json()
        print(f"Updated {len(result['result']['updated_fields'])} fields")

asyncio.run(patch_conversation_meta())
```

---

## Related Documentation

- [GroupChatFormat Specification](../../data_format/group_chat/group_chat_format.md)
- [Memory API Testing Guide](../dev_docs/memory_api_testing_guide.md)
- [run_memorize.py Usage Guide](../dev_docs/run_memorize_usage.md)

