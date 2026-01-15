# EverMemOS 记忆类型：Episode / Foresight / EventLog

---

## 1. 整体架构概览

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                              原始对话数据流                                     │
│                                   ↓                                           │
│                          ┌───────────────┐                                    │
│                          │   MemCell     │  ← 边界检测划分的记忆单元              │
│                          │  (记忆单元)    │                                    │
│                          └───────┬───────┘                                    │
│                                  │                                            │
│   ┌───────────────┬──────────────┼───────────────┬───────────────┐            │
│   │               │              │               │               │            │
│   ↓               ↓              ↓               ↓               ↓            │
│ ┌────────────────┐ ┌────────────────┐  ┌─────────────┐ ┌─────────────┐        │
│ │ Group Episode  │ │Personal Episode│  │  Foresight  │ │  EventLog   │        │
│ │  (所有场景)     │ │  (仅群聊场景)    │  │  (仅助手场景) │ │ (仅助手场景) │        │ 
│ └────────────────┘ └────────────────┘  └─────────────┘ └─────────────┘        │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 各记忆类型详解

### 2.1 MemCell（记忆单元）

**定义**：通过边界检测（Boundary Detection）从连续对话中划分出的最小记忆容器。

**核心字段**：

| 字段 | 说明 |
|------|------|
| `event_id` | 唯一标识符 |
| `original_data` | 原始对话消息列表 |
| `timestamp` | 记忆单元时间戳 |
| `summary` | 对话摘要 |
| `participants` | 参与者列表 |
| `group_id` | 群组ID |

**业务意义**：
- 是所有后续记忆提取的**数据源头**
- 边界检测确保每个 MemCell 代表一个完整的对话主题/事件
- **一对多关系**：每个 MemCell 可产生多条下游记忆（Episode、Foresight、EventLog），但每条下游记忆**只会来自唯一的一个 MemCell**

---

### 2.2 Episode（情节记忆）

**定义**：从 MemCell 中提取的叙事性记忆，是对一段对话的**高层语义理解**。

**类型**：

| 类型 | user_id | 说明 |
|------|---------|------|
| Group Episode | `None` | 全局/第三人称视角的客观叙述 |
| Personal Episode | 具体用户ID | 特定用户视角的主观叙述 |

**核心字段**：

| 字段 | 说明 |
|------|------|
| `subject` | 标题/主题 |
| `summary` | 摘要 |
| `episode` | 完整的情节叙述内容 |
| `memcell_event_id_list` | 来源 MemCell 的 ID 列表 |

**业务意义**：
- 为 AI 提供**人类可读的对话理解**
- 用于需要上下文理解的场景（如对话续写、背景介绍）

---

### 2.3 Foresight（前瞻记忆）

> ⚠️ **注意**：Foresight 仅在 **助手场景** 中提取，群聊不提取。

**定义**：基于对话内容直接生成的**未来预测**，预测对话对用户后续生活和决策的潜在影响。直接从 MemCell 提取。

**核心字段**：

| 字段 | 说明 |
|------|------|
| `foresight` | 预测内容 |
| `evidence` | 支撑该预测的证据 |
| `start_time` | 预测有效起始时间 |
| `end_time` | 预测有效结束时间 |
| `duration_days` | 预测有效天数 |
| `parent_type` | 关联的记忆类型(Memcell/Episode)（用于链接）|
| `parent_id` | 关联的 记忆 ID（用于链接）|

**业务意义**：
- 支持**主动推送**场景（如：提醒用户即将到来的事件）
- 支持**智能预判**场景（如：预判用户可能的后续需求）
- 带有**时效性**，可根据 `start_time` / `end_time` 过滤

**示例**：
```json
{
  "foresight": "用户可能需要在下周三之前完成项目报告",
  "evidence": "对话中提到'下周三是 deadline'",
  "start_time": "2024-03-10",
  "end_time": "2024-03-17",
  "duration_days": 7
}
```

---

### 2.4 EventLog（事件日志）

> ⚠️ **注意**：EventLog 仅在 **助手场景** 中提取，群聊不提取。

**定义**：直接从 MemCell的对话中提取的**原子化事实**，每个事实是独立的、可检索的最小语义单元。

> ⚠️ **重要**：LLM 提取时返回一个包含多个 atomic_fact 的列表，但**存入数据库时每个 atomic_fact 会拆分成独立的一条 `EventLogRecord` 记录**。

**数据库字段**（`event_log_records` Collection，每条记录一个 fact）：

| 字段 | 说明 |
|------|------|
| `atomic_fact` | **单条**原子事实（`str`） |
| `vector` | 该原子事实的向量表示 |
| `parent_type` | 关联的记忆类型(Memcell/Episode)（用于链接）|
| `parent_id` | 关联的记忆 ID（用于链接）|
| `timestamp` | 事件发生时间 |
| `user_id` / `group_id` | 归属信息 |

**业务意义**：
- 支持**精确检索**场景（向量检索 + BM25 关键词检索）
- 每个 atomic_fact **独立存储、独立向量化**，实现细粒度召回
- 适合回答「谁在什么时候做了什么事」类型的问题

**示例**（3 条数据库记录）：
```json
// 记录1
{ "atomic_fact": "张三提出了新的产品设计方案", "parent_type": "memcell", "parent_id": "mc_001", ... }
// 记录2
{ "atomic_fact": "李四表示同意该方案的技术可行性", "parent_type": "memcell", "parent_id": "mc_001", ... }
// 记录3  
{ "atomic_fact": "团队决定下周一开始原型开发", "parent_type": "memcell", "parent_id": "mc_001", ... }
```

---

## 3. 数量比例关系

### 3.1 基础比例

| 层级 | 数据库记录数 | 说明 |
|------|-------------|------|
| MemCell | 1 | 一次边界检测产生一个 |
| Episode | 1（助手）/ 1+N（群聊）| 助手场景：1个 Group（复制给用户）；群聊：1个 Group + N个 Personal |
| Foresight | ~10 | **仅助手场景**，每个 MemCell 产生 4-10 条预测 |
| EventLog | **M** | **仅助手场景**，M = atomic_fact 数（通常 5-15） |

### 3.2 典型场景示例

**场景A：助手对话（1 对 1）**
```
MemCell: 1
├── Group Episode: 1 (复制给用户)
├── Foresight: ~10 条 (从 MemCell 直接提取)
├── EventLog: ~8 条 (从 MemCell 直接提取)
└── 数据库总记录数: 1 Episode + 10 Foresight + 8 EventLog = 19
```

**场景B：群聊（3 人参与）**
```
MemCell: 1
├── Group Episode: 1
├── Personal Episode (UserA): 1
├── Personal Episode (UserB): 1
├── Personal Episode (UserC): 1
├── Foresight: 0 (群聊不提取)
├── EventLog: 0 (群聊不提取)
└── 数据库总记录数: 4 Episodes
```

> 💡 **注意**：Foresight 和 EventLog 仅在助手（1对1）场景提取，以提高效率。

---

## 4. 检索场景适配

| 检索需求 | 推荐记忆类型 | 检索方式 |
|----------|-------------|----------|
| 需要完整上下文 | Episode | 向量检索 + BM25 |
| 精确事实查询 | EventLog | 向量检索原子事实 |
| 未来事件提醒 | Foresight | 时间范围过滤 + 向量检索 |
| 用户画像补充 | Profile | 直接查询（非本文档范围） |

---

## 5. 存储与索引

### 5.1 持久化存储

| 记忆类型 | MongoDB Collection | 说明 |
|----------|-------------------|------|
| MemCell | `memcells` | 原始数据存储 |
| Episode | `episodic_memories` | 情节记忆存储 |
| Foresight | `foresight_records` | 前瞻记忆存储 |
| EventLog | `event_log_records` | 事件日志存储 |

### 5.2 搜索索引

| 记忆类型 | Elasticsearch | Milvus |
|----------|---------------|--------|
| Episode | ✅ 全文检索 | ✅ 向量检索 |
| Foresight | ✅ 全文检索 | ✅ 向量检索 |
| EventLog | ✅ 全文检索 | ✅ 向量检索（每个 fact 独立向量） |

---

## 6. 代码入口

### 6.1 提取入口

```python
# 主流程入口
biz_layer/mem_memorize.py::memorize()
    └── process_memory_extraction()
        ├── _extract_episodes()      # Episode 提取（所有场景）
        ├── _extract_foresights()    # Foresight 提取（仅助手场景）
        └── _extract_event_logs()    # EventLog 提取（仅助手场景）
```

### 6.2 各类型提取器

```
memory_layer/memory_extractor/
├── episode_memory_extractor.py    # EpisodeMemoryExtractor
├── foresight_extractor.py         # ForesightExtractor
└── event_log_extractor.py         # EventLogExtractor
```

### 6.3 检索入口

```python
# 统一检索入口
agentic_layer/memory_manager.py::MemoryManager
    ├── retrieve_mem()              # 标准检索
    ├── retrieve_lightweight()      # 轻量级检索（RRF/Embedding/BM25）
    └── retrieve_agentic()          # Agentic 多轮检索
```

---

## 7. 常见问题

**Q1：为什么 Episode 要区分 Group 和 Personal？**
- Group Episode 用于客观记录「发生了什么」
- Personal Episode 用于记录「对特定用户意味着什么」
- 在检索时可根据 user_id 过滤出用户关心的内容

**Q2：Foresight 的 10 条是固定的吗？**
- 代码中设定为提取 4-10 条预测
- 实际可能少于 10 条（LLM 输出不足时会 warning 但不会 retry）
- **仅在助手场景提取**（群聊不提取）

**Q3：EventLog 数据库记录数如何确定？**
- LLM 根据对话内容提取 atomic_fact 列表
- **每个 atomic_fact 单独存为一条 `EventLogRecord`**
- 通常每个 MemCell 产生 5-15 条 EventLog 记录，取决于对话复杂度
- **仅在助手场景提取**（群聊不提取）

**Q4：如何选择检索哪种记忆类型？**
- 需要语境理解 → Episode
- 需要精确事实 → EventLog
- 需要时效性预测 → Foresight

---

## 8. 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| v1.0 | 2024-12 | 初版文档 |
| v1.1 | 2025-01 | Foresight/EventLog 改为从 MemCell 直接提取（仅助手场景）|

