# Memory API 文档

## 目录

- [概述](#概述)
- [主要特性](#主要特性)
- [接口说明](#接口说明)
  - [POST `/api/v1/memories` - 存储单条消息记忆](#post-apiv1memories)
- [使用场景](#使用场景)
  - [1. 实时消息流处理](#1-实时消息流处理)
  - [2. 聊天机器人集成](#2-聊天机器人集成)
  - [3. 消息队列消费](#3-消息队列消费)
- [使用示例](#使用示例)
  - [使用 curl 调用](#使用-curl-调用)
  - [使用 Python 代码调用](#使用-python-代码调用)
  - [使用 run_memorize.py 脚本](#使用-run_memorizepy-脚本)
- [常见问题](#常见问题)
- [架构说明](#架构说明)
  - [数据流](#数据流)
  - [核心组件](#核心组件)
- [记忆查询接口](#记忆查询接口)
  - [GET `/api/v1/memories` - 获取用户记忆](#get-apiv1memories)
  - [GET `/api/v1/memories/search` - 检索相关记忆](#get-apiv1memoriessearch)
- [对话元数据管理](#对话元数据管理)
  - [POST `/api/v1/memories/conversation-meta` - 保存对话元数据](#post-apiv1memoriesconversation-meta)
  - [PATCH `/api/v1/memories/conversation-meta` - 局部更新对话元数据](#patch-apiv1memoriesconversation-meta)
- [相关文档](#相关文档)

---

## 概述

Memory API 提供了专门用于处理群聊记忆的接口，采用简单直接的消息格式，无需任何预处理或格式转换。

## 主要特性

- ✅ **简单直接**：采用最简单的单条消息格式，无需复杂的数据结构
- ✅ **无需转换**：不需要任何格式转换或适配
- ✅ **逐条处理**：实时处理每条消息，适合消息流场景
- ✅ **集中式适配**：所有格式转换逻辑集中在 `group_chat_converter.py`，保持单一职责
- ✅ **详细错误信息**：提供清晰的错误提示和数据统计

## 接口说明

### POST `/api/v1/memories`

逐条存储单条群聊消息记忆

#### 请求格式

**Content-Type**: `application/json`

**请求体**：简单直接的单条消息格式（无需预转换）

```json
{
  "group_id": "group_123",
  "group_name": "项目讨论组",
  "message_id": "msg_001",
  "create_time": "2025-01-15T02:00:00Z",
  "sender": "user_001",
  "sender_name": "张三",
  "content": "今天讨论下新功能的技术方案",
  "refer_list": ["msg_000"]
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| group_id | string | 否 | 群组ID |
| group_name | string | 否 | 群组名称 |
| message_id | string | 是 | 消息唯一标识 |
| create_time | string | 是 | 消息创建时间（ISO 8601格式） |
| sender | string | 是 | 发送者用户ID |
| sender_name | string | 否 | 发送者名称（默认使用 sender） |
| content | string | 是 | 消息内容 |
| refer_list | array | 否 | 引用的消息ID列表 |

#### 响应格式

**成功响应 (200 OK)**

根据记忆提取的状态，响应会有两种形式：

**1. 已提取记忆（extracted）** - 当消息触发了边界检测，成功提取了记忆：

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
        "content": "用户讨论了新功能的技术方案"
      }
    ],
    "count": 1,
    "status_info": "extracted"
  }
}
```

**2. 消息已累积（accumulated）** - 当消息已存储但未触发边界检测：

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

**字段说明**：
- `status_info`: 处理状态，`extracted` 表示已提取记忆，`accumulated` 表示消息已累积等待边界检测
- `saved_memories`: 已保存的记忆列表，当 `status_info` 为 `accumulated` 时为空数组
- `count`: 保存的记忆数量

**错误响应 (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "数据格式错误：缺少必需字段 message_id",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v1/memories"
}
```

**错误响应 (500 Internal Server Error)**

```json
{
  "status": "failed",
  "code": "SYSTEM_ERROR",
  "message": "存储记忆失败，请稍后重试",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v1/memories"
}
```

---

## 使用场景

### 1. 实时消息流处理

适用于处理来自聊天应用的实时消息流，每收到一条消息就立即存储。

**示例**：
```json
{
  "group_id": "group_123",
  "group_name": "项目讨论组",
  "message_id": "msg_001",
  "create_time": "2025-01-15T02:00:00Z",
  "sender": "user_001",
  "sender_name": "张三",
  "content": "今天讨论下新功能的技术方案",
  "refer_list": []
}
```

### 2. 聊天机器人集成

聊天机器人接收到用户消息后，可以直接调用 Memory API 存储记忆。

**示例**：
```json
{
  "group_id": "bot_conversation_123",
  "group_name": "与AI助手的对话",
  "message_id": "bot_msg_001",
  "create_time": "2025-01-15T02:05:00Z",
  "sender": "user_456",
  "sender_name": "李四",
  "content": "帮我总结下今天的会议内容",
  "refer_list": []
}
```

### 3. 消息队列消费

从消息队列（如 Kafka）消费消息时，可以逐条调用 Memory API 处理。

**Kafka 消费示例**：
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

# Kafka 消费者
consumer = KafkaConsumer('chat_messages')
for msg in consumer:
    asyncio.run(process_message(msg.value))
```

---

## 使用示例

### 使用 curl 调用

```bash
curl -X POST http://localhost:1995/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "group_name": "项目讨论组",
    "message_id": "msg_001",
    "create_time": "2025-01-15T02:00:00Z",
    "sender": "user_001",
    "sender_name": "张三",
    "content": "今天讨论下新功能的技术方案",
    "refer_list": []
  }'
```

### 使用 Python 代码调用

```python
import httpx
import asyncio

async def call_memory_api():
    # 简单直接的单条消息格式
    message_data = {
        "group_id": "group_123",
        "group_name": "项目讨论组",
        "message_id": "msg_001",
        "create_time": "2025-01-15T02:00:00Z",
        "sender": "user_001",
        "sender_name": "张三",
        "content": "今天讨论下新功能的技术方案",
        "refer_list": []
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v1/memories",
            json=message_data
        )
        result = response.json()
        print(f"保存了 {result['result']['count']} 条记忆")

asyncio.run(call_memory_api())
```

### 使用 run_memorize.py 脚本

对于 GroupChatFormat 格式的 JSON 文件，可以使用 `run_memorize.py` 脚本批量处理：

```bash
# 存储记忆
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories

# 仅验证格式
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --validate-only
```

---

## 常见问题

### 1. 如何处理带引用的消息？

使用 `refer_list` 字段指定引用的消息ID列表：

```json
{
  "message_id": "msg_002",
  "content": "我同意你的方案",
  "refer_list": ["msg_001"]
}
```

### 3. group_id 和 group_name 是必需的吗？

不是必需的，但**强烈推荐提供**：
- `group_id` 用于标识群组，方便后续检索
- `group_name` 用于显示和理解，提升可读性

### 4. 如何处理私聊消息？

私聊消息可以不提供 `group_id`，或者使用特殊的私聊ID：

```json
{
  "group_id": "private_user001_user002",
  "group_name": "与张三的私聊",
  "message_id": "private_msg_001",
  "create_time": "2025-01-15T02:00:00Z",
  "sender": "user_001",
  "sender_name": "张三",
  "content": "你好，最近怎么样？",
  "refer_list": []
}
```

### 5. 如何处理消息时间？

`create_time` 必须使用 ISO 8601 格式，支持带时区：

```json
{
  "create_time": "2025-01-15T02:00:00Z"  // 带时区
}
```

或不带时区（默认使用 UTC）：

```json
{
  "create_time": "2025-01-15T10:00:00"  // UTC
}
```

### 6. 如何批量处理历史消息？

使用 `run_memorize.py` 脚本：

1. 准备 GroupChatFormat 格式的 JSON 文件
2. 运行脚本，脚本会自动逐条调用 Memory API

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v1/memories
```

### 7. 接口调用频率有限制吗？

目前没有硬性限制，但建议：
- 实时场景：每秒不超过 100 次请求
- 批量导入：建议每条消息间隔 0.1 秒

### 8. 如何处理错误？

接口会返回详细的错误信息：

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "缺少必需字段: message_id"
}
```

建议在客户端实现重试机制，对于 5xx 错误可以重试 3 次。

---

## 架构说明

### 数据流

```
客户端
  ↓
  │ 简单直接的单条消息格式
  ↓
Memory Controller (memory_controller.py)
  ↓
  │ 调用 group_chat_converter.py
  ↓
格式转换 (convert_simple_message_to_memorize_input)
  ↓
  │ 内部格式
  ↓
Memory Manager (memory_manager.py)
  ↓
  │ 记忆存储
  ↓
数据库 / 向量库
```

### 核心组件

1. **Memory Controller** (`memory_controller.py`)
   - 接收简单直接的单条消息
   - 调用 converter 进行格式转换
   - 调用 memory_manager 存储记忆

2. **Group Chat Converter** (`group_chat_converter.py`)
   - 集中式适配层
   - 负责所有格式转换逻辑
   - 保持单一职责

3. **Memory Manager** (`memory_manager.py`)
   - 记忆提取和存储
   - 向量化
   - 持久化

---

## 记忆查询接口

### GET `/api/v1/memories`

通过 KV 方式获取用户的核心记忆数据。

#### 功能说明

- 根据用户ID直接获取存储的核心记忆
- 支持多种记忆类型：基础记忆、用户画像、偏好设置等
- 支持分页和排序
- 适用于需要快速获取用户固定记忆集合的场景

#### 请求参数（Query Parameters）

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|-----|------|------|--------|------|
| user_id | string | 是 | - | 用户ID |
| memory_type | string | 否 | "multiple" | 记忆类型，可选值：`base_memory`、`profile`、`preference`、`episode_memory`、`multiple` |
| limit | integer | 否 | 40 | 返回记忆的最大数量 |
| offset | integer | 否 | 0 | 分页偏移量 |
| sort_by | string | 否 | - | 排序字段 |
| sort_order | string | 否 | "desc" | 排序方向，可选值：`asc`、`desc` |

**记忆类型说明**：
- `base_memory`: 基础记忆，用户的基本信息和常用数据
- `profile`: 用户画像，包含用户的特征和属性
- `preference`: 用户偏好，包含用户的喜好和设置
- `episode_memory`: 情景记忆摘要
- `multiple`: 多类型（默认），包含 base_memory、profile、preference

#### 响应格式

**成功响应 (200 OK)**

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
        "content": "用户喜欢喝咖啡",
        "summary": "咖啡偏好"
      },
      {
        "memory_type": "profile",
        "user_id": "user_123",
        "timestamp": "2024-01-14T09:20:00",
        "content": "用户是一名软件工程师"
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

#### 使用示例

**使用 curl**：

```bash
curl -X GET "http://localhost:1995/api/v1/memories?user_id=user_123&memory_type=multiple&limit=20" \
  -H "Content-Type: application/json"
```

**使用 Python**：

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
        print(f"获取了 {len(result['result']['memories'])} 条记忆")

asyncio.run(fetch_memories())
```

---

### GET `/api/v1/memories/search`

基于查询文本使用关键词、向量或混合方法检索相关的记忆数据。

#### 功能说明

- 根据查询文本查找最相关的记忆
- 支持关键词（BM25）、向量相似度、混合检索三种方法
- 支持时间范围过滤
- 返回结果按群组组织，并包含相关性评分
- 适用于需要精确匹配或语义检索的场景

#### 请求格式

**Content-Type**: `application/json`

**请求体**：

```json
{
  "user_id": "user_123",
  "query": "咖啡偏好",
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

**字段说明**：

| 字段 | 类型 | 必需 | 默认值 | 说明 |
|-----|------|------|--------|------|
| user_id | string | 是 | - | 用户ID |
| query | string | 否 | - | 查询文本 |
| retrieve_method | string | 否 | "keyword" | 检索方法，可选值：`keyword`、`vector`、`hybrid` |
| top_k | integer | 否 | 40 | 返回的最大结果数 |
| start_time | string | 否 | - | 时间范围起点（ISO 8601格式） |
| end_time | string | 否 | - | 时间范围终点（ISO 8601格式） |
| memory_types | array | 否 | [] | 要检索的记忆类型列表 |
| filters | object | 否 | {} | 额外的过滤条件 |
| radius | float | 否 | 0.6 | 向量检索时的相似度阈值（仅对 vector 和 hybrid 方法有效） |

**检索方法说明**：
- `keyword`: 基于关键词的 BM25 检索，适合精确匹配，速度快（默认方法）
- `vector`: 基于语义向量的相似度检索，适合模糊查询和语义相似查询
- `hybrid`: 混合检索策略，结合关键词和向量检索的优势（推荐）

#### 响应格式

**成功响应 (200 OK)**

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
            "summary": "讨论了咖啡偏好",
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

**返回结果说明**：
- `memories`: 记忆列表，按群组（group）组织
- `scores`: 每条记忆的相关性得分
- `importance_scores`: 群组重要性得分，用于排序
- `total_count`: 总记忆数
- `has_more`: 是否还有更多结果

#### 使用示例

**使用 curl**：

```bash
curl -X GET http://localhost:1995/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "query": "咖啡偏好",
    "retrieve_method": "keyword",
    "top_k": 10
  }'
```

**使用 Python**：

```python
import httpx
import asyncio

async def search_memories():
    search_data = {
        "user_id": "user_123",
        "query": "咖啡偏好",
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
        print(f"检索到 {result['result']['total_count']} 条记忆")

asyncio.run(search_memories())
```

---

## 对话元数据管理

### POST `/api/v1/memories/conversation-meta`

保存对话的元数据信息，包括场景、参与者、标签等。

#### 请求格式

**Content-Type**: `application/json`

**请求体**：

```json
{
  "version": "1.0",
  "scene": "group_chat",
  "scene_desc": "项目团队讨论",
  "name": "项目讨论组",
  "description": "新功能开发的技术讨论",
  "group_id": "group_123",
  "created_at": "2025-01-15T02:00:00Z",
  "default_timezone": "UTC",
  "user_details": {
    "user_001": {
      "full_name": "张三",
      "role": "developer",
      "extra": {"department": "工程部"}
    },
    "user_002": {
      "full_name": "李四",
      "role": "designer",
      "extra": {"department": "设计部"}
    }
  },
  "tags": ["工作", "技术"]
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|-----|------|------|------|
| version | string | 是 | 元数据版本 |
| scene | string | 是 | 场景标识（如 "group_chat"） |
| scene_desc | string | 是 | 场景描述 |
| name | string | 是 | 对话名称 |
| description | string | 是 | 对话描述 |
| group_id | string | 是 | 群组唯一标识 |
| created_at | string | 是 | 对话创建时间（ISO 8601 格式） |
| default_timezone | string | 是 | 默认时区 |
| user_details | object | 是 | 参与者详情 |
| tags | array | 否 | 标签列表 |

#### 响应格式

**成功响应 (200 OK)**

```json
{
  "status": "ok",
  "message": "对话元数据保存成功",
  "result": {
    "id": "507f1f77bcf86cd799439011",
    "group_id": "group_123",
    "scene": "group_chat",
    "name": "项目讨论组",
    "version": "1.0",
    "created_at": "2025-01-15T02:00:00Z",
    "updated_at": "2025-01-15T02:00:00Z"
  }
}
```

**注意**：该接口使用 upsert 行为，如果 `group_id` 已存在，则会更新整个记录。

---

### PATCH `/api/v1/memories/conversation-meta`

局部更新对话元数据，只更新提供的字段。

#### 请求格式

**Content-Type**: `application/json`

**请求体**（只需提供需要更新的字段）：

```json
{
  "group_id": "group_123",
  "name": "新的对话名称",
  "tags": ["标签1", "标签2"]
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|-----|------|------|------|
| group_id | string | 是 | 要更新的群组ID |
| name | string | 否 | 新的对话名称 |
| description | string | 否 | 新的描述 |
| scene_desc | string | 否 | 新的场景描述 |
| tags | array | 否 | 新的标签列表 |
| user_details | object | 否 | 新的用户详情（会完整替换现有的 user_details） |
| default_timezone | string | 否 | 新的默认时区 |

**可更新的字段**：
- `name`: 对话名称
- `description`: 对话描述
- `scene_desc`: 场景描述
- `tags`: 标签列表
- `user_details`: 用户详情（会完整替换现有的 user_details）
- `default_timezone`: 默认时区

**不可修改的字段**（不能通过 PATCH 修改）：
- `version`: 元数据版本
- `scene`: 场景标识
- `group_id`: 群组ID
- `conversation_created_at`: 对话创建时间

#### 响应格式

**成功响应 (200 OK)**

```json
{
  "status": "ok",
  "message": "对话元数据更新成功，共更新 2 个字段",
  "result": {
    "id": "507f1f77bcf86cd799439011",
    "group_id": "group_123",
    "scene": "group_chat",
    "name": "新的对话名称",
    "updated_fields": ["name", "tags"],
    "updated_at": "2025-01-15T02:30:00Z"
  }
}
```

**错误响应 (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "缺少必需字段 group_id",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v1/memories/conversation-meta"
}
```

**错误响应 (404 Not Found)**

```json
{
  "status": "failed",
  "code": "RESOURCE_NOT_FOUND",
  "message": "找不到指定的对话元数据: group_123",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v1/memories/conversation-meta"
}
```

#### 使用示例

**使用 curl**：

```bash
# 局部更新对话元数据
curl -X PATCH http://localhost:1995/api/v1/memories/conversation-meta \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "name": "新的对话名称",
    "tags": ["更新", "标签"]
  }'
```

**使用 Python**：

```python
import httpx
import asyncio

async def patch_conversation_meta():
    update_data = {
        "group_id": "group_123",
        "name": "新的对话名称",
        "tags": ["更新", "标签"]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.patch(
            "http://localhost:1995/api/v1/memories/conversation-meta",
            json=update_data
        )
        result = response.json()
        print(f"更新了 {len(result['result']['updated_fields'])} 个字段")

asyncio.run(patch_conversation_meta())
```

---

## 相关文档

- [GroupChatFormat 格式规范](../../data_format/group_chat/group_chat_format.md)
- [Memory API 测试指南](../dev_docs/memory_api_testing_guide.md)
- [run_memorize.py 使用指南](../dev_docs/run_memorize_usage.md)
