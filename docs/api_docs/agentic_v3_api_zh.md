# Agentic Layer V3 API 文档

## 概述

Agentic Layer V3 在 `/api/v3/agentic` 下提供三个 FastAPI 端点，用于写入群聊记忆与检索历史。所有端点仅接受 `application/json`，成功响应统一为：

```
{
  "status": "ok",
  "message": "<摘要>",
  "result": { ... }
}
```

当校验失败或出现异常时，`status` 会变为 `failed`。

## 端点一览

| 端点 | 作用 |
|------|------|
| `POST /api/v3/agentic/memorize` | 写入单条聊天消息，提取 memcells |
| `POST /api/v3/agentic/retrieve_lightweight` | 低延迟混合检索（Embedding + BM25 + RRF） |
| `POST /api/v3/agentic/retrieve_agentic` | LLM 驱动的多轮智能检索（含 Rerank） |

---

## POST `/api/v3/agentic/memorize`

接收未转换的原始消息。提供 `group_id` 时，原始请求还会写入 Redis `chat_history:{group_id}`（左推入，TTL 24h），用于后续边界检测。

### 请求字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `group_id` | string | 否 | 标识群组，开启 Redis 累积 |
| `group_name` | string | 否 | 群组名称 |
| `message_id` | string | 是 | 单条消息唯一 ID |
| `create_time` | string | 是 | ISO 8601 时间，可含时区 |
| `sender` | string | 是 | 发送者用户 ID |
| `sender_name` | string | 否 | 缺省等于 `sender` |
| `content` | string | 是 | 文本内容 |
| `refer_list` | array\<string> | 否 | 被引用的消息 ID |

### 成功响应

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
        "content": "用户讨论了新功能方案"
      }
    ],
    "count": 1,
    "status_info": "extracted"
  }
}
```

当边界尚未触发时，`status_info` 为 `accumulated`，`saved_memories` 为空，但原始消息已安全入队。

### 异常

- `400 Bad Request`：字段缺失或格式错误，返回 `{"detail": "..."}`。
- `500 Internal Server Error`：转换或存储流程发生未捕获错误。

---

## POST `/api/v3/agentic/retrieve_lightweight`

同时执行向量检索与 BM25，使用 RRF 融合并返回结果。

### 请求字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | 是* | — | `data_source="profile"` 时可为空 |
| `user_id` | string | 否 | `null` | 过滤个人记忆 |
| `group_id` | string | 否 | `null` | 过滤群聊记忆 |
| `time_range_days` | integer | 否 | `365` | 滑动时间窗口 |
| `top_k` | integer | 否 | `20` | 返回条数上限 |
| `retrieval_mode` | string | 否 | `rrf` | `rrf` / `embedding` / `bm25` |
| `data_source` | string | 否 | `episode` | `episode` / `event_log` / `semantic_memory` / `profile`，`memcell` 会映射为 `episode` |
| `memory_scope` | string | 否 | `all` | `all` / `personal` / `group` |
| `current_time` | string | 否 | `null` | `YYYY-MM-DD`，语义记忆过滤时间 |
| `radius` | float | 否 | `null` | 余弦相似度阈值 `[-1, 1]` |

\* 当 `data_source="profile"` 时，必须同时提供 `user_id` 与 `group_id`，即使 `query` 为空。

### 成功响应

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

### 异常

- `400 Bad Request`：`current_time` 格式错误、缺少 `query`、`profile` 场景缺少 ID 等。
- `500 Internal Server Error`：检索流程抛出未捕获异常。

---

## POST `/api/v3/agentic/retrieve_agentic`

执行 agentic 流程：首轮混合检索 → Rerank → LLM 判断是否充分 → 生成改写查询 → 第二轮检索 → 最终 Rerank。

### 请求字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | string | 是 | — | 自然语言查询 |
| `user_id` | string | 否 | `null` | 个人过滤条件 |
| `group_id` | string | 否 | `null` | 群聊过滤条件 |
| `time_range_days` | integer | 否 | `365` | 时间窗口 |
| `top_k` | integer | 否 | `20` | 最终返回条数 |
| `llm_config.api_key` | string | 是* | — | OpenAI 兼容 API Key |
| `llm_config.base_url` | string | 否 | `https://openrouter.ai/api/v1` | 自定义 OpenAI 兼容地址 |
| `llm_config.model` | string | 否 | `qwen/qwen3-235b-a22b-2507` | LLMProvider 使用的模型标识 |

\* 可从 `llm_config.api_key`、`OPENROUTER_API_KEY` 或 `OPENAI_API_KEY` 提供，缺失时返回 `400`。

### 成功响应

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

### 异常

- `400 Bad Request`：缺少 `query`、缺少 API Key 或参数格式错误。
- `500 Internal Server Error`：LLM 调用或记忆子系统返回异常。

---

## 错误载荷

服务通过 `HTTPException` 统一抛错，网关会序列化为：

```
{
  "status": "failed",
  "code": "<ErrorCode>",
  "message": "<错误说明>",
  "timestamp": "<ISO 8601>",
  "path": "<请求路径>"
}
```

对于 `5xx`，建议客户端按需实现幂等重试。*** End Patch***}github.com to=functions.apply_patch

