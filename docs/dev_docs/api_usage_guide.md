# API Usage Guide

This document provides detailed instructions on how to use MemSys API interfaces to store and retrieve memory data.

## 📋 Table of Contents

- [API Overview](#api-overview)
- [Memory Storage APIs](#memory-storage-apis)
  - [V1 Memory API](#v3-agentic-api)
  - [V1 Memory API](#v1-memory-api)
  - [API Selection Guide](#api-selection-guide)
- [Group Chat Data Format](#group-chat-data-format)
- [Using Scripts to Store Memories](#using-scripts-to-store-memories)
- [API Call Examples](#api-call-examples)

## 🔍 API Overview

MemSys provides two standardized API interfaces for storing memories:

### Available APIs

| API Type | Endpoint | Features | Recommended Use Case |
|---------|---------|------|---------|
| **V1 Memory API** | `/api/v1/memories` | Memory Storage + Intelligent Retrieval | Complete application scenarios requiring retrieval features |

### API Comparison

| Feature | V1 Memory API | V1 Memory API |
|-----|---------------|--------------|
| Store Single Message | ✅ Supported | ✅ Supported |
| Message Format | Simple direct single message format | Simple direct single message format |
| Intelligent Retrieval | ✅ Supported (Lightweight + Agentic) | ❌ Not Supported |
| Session Metadata Management | ✅ Supported | ✅ Supported (with PATCH updates) |
| Use Case | Complete memory system (storage + retrieval) | Pure memory storage system |

**Important Note**: Both APIs use identical storage formats, so you can choose based on your needs. If you need retrieval functionality, we recommend using V1 Memory API for complete feature support.

---

## 🚀 Memory Storage APIs

### V1 Memory API

Recommended for scenarios requiring complete functionality (storage + retrieval).

#### Endpoint

```
POST /api/v1/memories
```

#### Features

- ✅ Simple direct single message format
- ✅ Supports lightweight retrieval (RRF fusion)
- ✅ Supports Agentic intelligent retrieval (LLM-assisted)
- ✅ Supports session metadata management

For detailed documentation, see: [Memory API Documentation](../api_docs/memory_api.md)

---

### V1 Memory API

Recommended for simple scenarios requiring only storage functionality.

#### Endpoint

```
POST /api/v1/memories
```

#### Features

- ✅ Simple direct single message format
- ✅ Focused on memory storage
- ✅ Supports session metadata management (with PATCH partial updates)

For detailed documentation, see: [Memory API Documentation](../api_docs/memory_api.md)

---

### API Selection Guide

**Use V1 Memory API (`/api/v1/memories`)** if:
- ✅ You need intelligent retrieval functionality
- ✅ You need to build a complete memory system (storage + retrieval)
- ✅ You want to use lightweight or Agentic retrieval modes

**Use V1 Memory API (`/api/v1/memories`)** if:
- ✅ You only need to store memories without retrieval
- ✅ You have your own retrieval solution
- ✅ You prefer a more concise dedicated storage interface

**Note**: Both APIs use identical data formats and underlying storage mechanisms. The main difference is that V1 API provides additional retrieval functionality.

---

## 📝 Memorize API Details

### Request Format (Common to Both APIs)

Both APIs use the same simple direct single message format:

```json
{
  "message_id": "msg_001",
  "create_time": "2025-02-01T10:00:00+00:00",
  "sender": "user_103",
  "sender_name": "Chen",
  "content": "Message content",
  "refer_list": [],
  "group_id": "group_001",
  "group_name": "Project Discussion Group"
}
```

### Field Descriptions

| Field | Type | Required | Description |
|------|------|------|------|
| `message_id` | string | Yes | Unique message identifier |
| `create_time` | string | Yes | Message creation time (ISO 8601 format) |
| `sender` | string | Yes | Sender ID |
| `sender_name` | string | No | Sender name (for readability) |
| `content` | string | Yes | Message content |
| `refer_list` | array | No | List of referenced messages |
| `group_id` | string | No | Group ID |
| `group_name` | string | No | Group name |

### Response Format

```json
{
  "code": 0,
  "message": "success",
  "result": {
    "count": 2,
    "saved_memories": [
      {
        "memory_id": "mem_001",
        "type": "episode",
        "content": "Extracted memory content"
      }
    ]
  }
}
```

### Call Examples

#### cURL

```bash
curl -X POST http://localhost:1995/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+00:00",
    "sender": "user_103",
    "sender_name": "Chen",
    "content": "We need to complete the product design this week",
    "group_id": "group_001",
    "group_name": "Project Discussion Group"
  }'
```

#### Python

```python
import httpx
import asyncio

async def store_memory():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v1/memories",
            json={
                "message_id": "msg_001",
                "create_time": "2025-02-01T10:00:00+00:00",
                "sender": "user_103",
                "sender_name": "Chen",
                "content": "We need to complete the product design this week",
                "group_id": "group_001",
                "group_name": "Project Discussion Group"
            }
        )
        print(response.json())

asyncio.run(store_memory())
```

#### JavaScript

```javascript
fetch('http://localhost:1995/api/v1/memories', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message_id: 'msg_001',
    create_time: '2025-02-01T10:00:00+00:00',
    sender: 'user_103',
    sender_name: 'Chen',
    content: 'We need to complete the product design this week',
    group_id: 'group_001',
    group_name: 'Project Discussion Group'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

**Using V1 Memory API:**

```javascript
fetch('http://localhost:1995/api/v1/memories', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message_id: 'msg_001',
    create_time: '2025-02-01T10:00:00+00:00',
    sender: 'user_103',
    sender_name: 'Chen',
    content: 'We need to complete the product design this week',
    group_id: 'group_001',
    group_name: 'Project Discussion Group'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 📁 Group Chat Data Format

MemSys defines a standardized group chat data format `GroupChatFormat` for storing and exchanging group chat conversation data.

### Format Overview

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "group_id": "group_001",
    "name": "Project Discussion Group",
    "default_timezone": "+00:00",
    "user_details": {
      "user_101": {
        "full_name": "Alex",
        "role": "Technical Lead",
        "department": "Engineering"
      }
    }
  },
  "conversation_list": [
    {
      "message_id": "msg_001",
      "create_time": "2025-02-01T10:00:00+00:00",
      "sender": "user_101",
      "sender_name": "Alex",
      "type": "text",
      "content": "Good morning everyone",
      "refer_list": []
    }
  ]
}
```

### Core Features

1. **Separated Metadata and Message List**
   - `conversation_meta`: Group chat metadata
   - `conversation_list`: Message list

2. **Centralized User Details**
   - All user information stored in `user_details`
   - Messages only need to reference user IDs

3. **Timezone-aware Timestamps**
   - Uses ISO 8601 format
   - Supports timezone information

4. **Flexible Message References**
   - Supports string references (message_id only)
   - Supports object references (complete message information)

### Detailed Documentation

For complete format specification, see: [Group Chat Format Specification](../../data_format/group_chat/group_chat_format.md)

## 🔧 Using Scripts to Store Memories

MemSys provides the `run_memorize.py` script for batch storing group chat data into the system. The script supports both API interfaces.

### Script Location

```
src/run_memorize.py
```

### Basic Usage

Run using the Bootstrap script with V1 API:

**Using V1 Memory API (Recommended, supports retrieval):**

```bash
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories
```

**Using V1 Memory API (Storage only):**

```bash
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories
```

### Command Line Arguments

| Argument | Required | Description |
|------|------|------|
| `--input` | Yes | Input group chat JSON file path (GroupChatFormat) |
| `--api-url` | No* | Memorize API address (*unless using --validate-only) |
| `--validate-only` | No | Only validate input file format without storing |

### Usage Examples

#### 1. Store Memories

**Using V1 Memory API:**

```bash
# Basic usage
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories

# Using relative path
uv run python src/bootstrap.py src/run_memorize.py \
  --input ../my_data/chat_history.json \
  --api-url http://localhost:1995/api/v1/memories

# Specifying remote server
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://api.example.com/api/v1/memories
```

**Using V1 Memory API:**

```bash
# Basic usage
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories

# Using relative path
uv run python src/bootstrap.py src/run_memorize.py \
  --input ../my_data/chat_history.json \
  --api-url http://localhost:1995/api/v1/memories
```

#### 2. Validate File Format

Validate file format before storing:

```bash
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --validate-only
```

### Script Workflow

1. **Validate Input File**
   - Check if JSON format is correct
   - Verify compliance with GroupChatFormat specification
   - Output data statistics

2. **Process Messages One by One**
   - Read each message from group chat file
   - Call API to store each message
   - Display processing progress and results

3. **Output Processing Results**
   - Number of successfully processed messages
   - Number of saved memories
   - Failed messages (if any)

### Output Example

```
🚀 Group Chat Memory Storage Script
======================================================================
📄 Input File: /path/to/data/group_chat.json
🔍 Validation Mode: No
🌐 API Address: http://localhost:1995/api/v1/memories
======================================================================

======================================================================
Validating Input File Format
======================================================================
Reading file: /path/to/data/group_chat.json
Validating GroupChatFormat...
✓ Format validation passed!

=== Data Statistics ===
Format Version: 1.0.0
Group Name: Project Discussion Group
Group ID: group_001
User Count: 5
Message Count: 20
Time Range: 2025-02-01T10:00:00+00:00 ~ 2025-02-01T18:30:00+00:00

======================================================================
Starting to Call Memorize API for Each Message
======================================================================
Group Name: Project Discussion Group
Group ID: group_001
Message Count: 20
API Address: http://localhost:1995/api/v1/memories

--- Processing Message 1/20 ---
  ✓ Successfully saved 2 memories

--- Processing Message 2/20 ---
  ✓ Successfully saved 1 memory

...

======================================================================
Processing Complete
======================================================================
✓ Successfully Processed: 20/20 messages
✓ Total Saved: 35 memories
```

## 📝 API Call Examples

### Complete Workflow

#### 1. Prepare Data File

Create a JSON file conforming to GroupChatFormat:

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "group_id": "project_team_001",
    "name": "Product Development Team",
    "default_timezone": "+00:00",
    "user_details": {
      "alice": {
        "full_name": "Alice Wang",
        "role": "Product Manager",
        "department": "Product"
      },
      "bob": {
        "full_name": "Bob Chen",
        "role": "Technical Lead",
        "department": "Engineering"
      }
    }
  },
  "conversation_list": [
    {
      "message_id": "msg_20250201_001",
      "create_time": "2025-02-01T09:00:00+00:00",
      "sender": "alice",
      "sender_name": "Alice Wang",
      "type": "text",
      "content": "Good morning! Let's discuss the new feature requirements today",
      "refer_list": []
    },
    {
      "message_id": "msg_20250201_002",
      "create_time": "2025-02-01T09:02:00+00:00",
      "sender": "bob",
      "sender_name": "Bob Chen",
      "type": "text",
      "content": "Sure, I've prepared some technical solutions",
      "refer_list": ["msg_20250201_001"]
    }
  ]
}
```

Save as `my_chat_data.json`.

#### 2. Validate File Format

```bash
uv run python src/bootstrap.py src/run_memorize.py \
  --input my_chat_data.json \
  --validate-only
```

#### 3. Start Service

Ensure MemSys service is running:

```bash
uv run python src/run.py
```

After service starts, visit http://localhost:1995/docs to verify API documentation is accessible.

#### 4. Store Memories

**Option A: Using V1 Memory API (Recommended)**

```bash
uv run python src/bootstrap.py src/run_memorize.py \
  --input my_chat_data.json \
  --api-url http://localhost:1995/api/v1/memories
```

**Option B: Using V1 Memory API**

```bash
uv run python src/bootstrap.py src/run_memorize.py \
  --input my_chat_data.json \
  --api-url http://localhost:1995/api/v1/memories
```

#### 5. Verify Storage Results

If using V1 Memory API, you can query stored memories through the retrieval interface (see [Memory API Documentation](../api_docs/memory_api.md) for specific query APIs).

### Error Handling

#### Format Validation Failed

```
✗ Format validation failed!
Please ensure input file conforms to GroupChatFormat specification
```

**Solution**:
- Check if JSON format is correct
- Refer to [Group Chat Format Specification](../../data_format/group_chat/group_chat_format.md)
- Ensure all required fields are filled

#### API Call Failed

```
✗ API call failed: 500
Response content: {"error": "Internal server error"}
```

**Solution**:
- Check if service is running normally
- View service logs to troubleshoot
- Verify API address is correct

#### Connection Timeout

```
✗ Processing failed: ReadTimeout
```

**Solution**:
- Check network connection
- Verify service address and port are correct
- Check firewall settings

## 🔗 Related Documentation

### API Documentation

- [Memory API Documentation](../api_docs/memory_api.md) - Complete V1 API documentation (storage + retrieval)
- [Memory API Documentation](../api_docs/memory_api.md) - Complete V1 Memory API documentation (focused on storage)

### Other Documentation

- [Group Chat Format Specification](../../data_format/group_chat/group_chat_format.md) - Detailed GroupChatFormat specification
- [Getting Started Guide](getting_started.md) - Environment setup and service startup
- [Agentic Retrieval Guide](agentic_retrieval_guide.md) - Intelligent retrieval features explained

## 💡 Best Practices

1. **Data Preparation**
   - Use standard GroupChatFormat
   - Ensure timestamps include timezone information
   - Provide complete user details

2. **Batch Processing**
   - For large number of messages, use script to process one by one
   - Add appropriate delays to avoid server pressure
   - Monitor processing progress and errors

3. **Error Recovery**
   - Log failed messages
   - Support resume from checkpoint
   - Regularly verify storage results

4. **Performance Optimization**
   - Set reasonable concurrency levels
   - Use batch interfaces (if available)
   - Monitor API response times

---

For questions, please refer to [FAQ](getting_started.md#faq) or submit an issue.

