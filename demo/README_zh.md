# Demo - EverMemOS 交互式示例

[English](README.md) | [简体中文](README_zh.md)

本目录包含展示 EverMemOS 核心功能的交互式演示。

## 📂 目录结构

```
demo/
├── chat_with_memory.py          # 🎯 主程序：记忆增强对话
├── extract_memory.py            # 🎯 主程序：对话记忆提取（HTTP API）
├── simple_demo.py               # 🎯 主程序：快速入门示例
│
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── memory_utils.py         # 共享工具函数
│   └── simple_memory_manager.py # 简单记忆管理器（HTTP API 封装）
│
├── ui/                          # UI 模块
│   ├── __init__.py
│   └── i18n_texts.py           # 国际化文本
│
├── chat/                        # 聊天系统组件
│   ├── __init__.py
│   ├── orchestrator.py         # 聊天应用编排器
│   ├── session.py              # 会话管理
│   ├── ui.py                   # 用户界面
│   └── selectors.py            # 语言/场景/群组选择器
│
├── tools/                       # 辅助工具
│   ├── clear_all_data.py       # 清理所有记忆数据
│   ├── resync_memcells.py      # 重新同步记忆单元
│   └── test_retrieval_comprehensive.py  # 检索测试工具
│
├── chat_history/                # 📁 输出：对话记录（自动生成）
│
├── README.md                    # 📖 文档（英文）
└── README_zh.md                 # 📖 文档（中文）
```

**说明**：
- 所有记忆数据存储到数据库（MongoDB、Elasticsearch、Milvus），不生成本地 `memcell_outputs/` 目录
- `extract_memory.py` 直接调用 HTTP API，无需复杂配置类
- 对话历史保存在 `chat_history/` 目录中

## 🎯 核心脚本

### 1. `simple_demo.py` - 快速入门示例 ⭐

**体验 EverMemOS 最简单的方式！** 只需 67 行代码就能演示完整的记忆工作流程。

**演示内容：**
- 💾 **存储**：通过 HTTP API 保存对话消息
- ⏳ **索引**：等待数据被索引（MongoDB、Elasticsearch、Milvus）
- 🔍 **搜索**：用自然语言查询检索相关记忆

**代码示例：**
```python
from demo.utils import SimpleMemoryManager

# 创建记忆管理器
memory = SimpleMemoryManager()

# 存储对话
await memory.store("I love playing soccer, often go to the field on weekends")
await memory.store("Soccer is a great sport! Which team do you like?", sender="Assistant")
await memory.store("I love Barcelona the most, Messi is my idol")

# 等待索引构建
await memory.wait_for_index(seconds=10)

# 搜索记忆
await memory.search("What sports does the user like?")
await memory.search("What is the user's favorite team?")
```

**运行方式：**

⚠️ **重要**：必须先启动 API 服务器！

```bash
# 终端 1：启动 API 服务器
uv run python src/bootstrap.py src/run.py --port 8001

# 终端 2：运行简单演示
uv run python src/bootstrap.py demo/simple_demo.py
```

**为什么选择这个演示？**
- ✅ 代码极简 - 几秒钟就能理解核心概念
- ✅ 完整工作流 - 存储 → 索引 → 检索
- ✅ 友好输出 - 每一步都有说明
- ✅ 真实 HTTP API - 使用与生产环境相同的 API

**依赖**: `utils/simple_memory_manager.py`（HTTP API 封装器）

### 2. `extract_memory.py` - 记忆提取

通过 HTTP API 批量处理对话数据并提取记忆。

**工作流程**：
- 清空所有已存在的记忆（确保干净的初始状态）
- 从 `data/` 目录加载对话文件（如 `data/assistant_chat_zh.json`）
- 逐条发送消息到 API 服务器 (`/api/v3/agentic/memorize`)
- 服务器端自动提取 MemCell、生成情节和画像
- 所有数据存储到数据库（MongoDB、Elasticsearch、Milvus）

**运行前提**：必须先启动 API 服务器 (`uv run python src/bootstrap.py src/run.py --port 8001`)

**依赖**: HTTP API、`clear_all_data` 工具

### 3. `chat_with_memory.py` - 记忆增强对话

通过命令行界面与具有记忆能力的 AI 智能体对话。

**功能特性**：
- 交互式语言选择（中文/英文）和场景选择（助手/群聊）
- 自动从 MongoDB 加载对话群组
- 灵活的检索模式选择（RRF/Embedding/BM25/Agentic）
- 实时显示检索到的记忆
- 对话历史自动保存

**运行前提**：必须先运行 `extract_memory.py` 提取记忆数据

**依赖**: `chat/` 模块、HTTP API

## 📦 支持模块

### 工具模块
- **`utils/simple_memory_manager.py`** - 简化的 HTTP API 封装器，用于 simple_demo
- **`utils/memory_utils.py`** - MongoDB 连接和通用工具函数

### UI 模块
- **`ui/i18n_texts.py`** - 双语界面文本资源（中文/英文）

### 核心组件
- **`chat/`** - 聊天系统实现（编排器、会话管理、界面、选择器）
- **`tools/`** - 辅助工具（数据清理、检索测试等）

## 🚀 快速开始

### 方式 A：超级简单模式（推荐新手）⭐

体验 EverMemOS 最快的方式！只需 2 个终端：

```bash
# 终端 1：启动 API 服务器（必需）
uv run python src/bootstrap.py src/run.py --port 8001

# 终端 2：运行简单演示
uv run python src/bootstrap.py demo/simple_demo.py
```

**发生了什么：**
1. 📝 存储 4 条对话消息
2. ⏳ 等待 10 秒建立索引（MongoDB → Elasticsearch → Milvus）
3. 🔍 用 3 个不同的查询搜索记忆
4. 📊 显示结果（相关度分数和说明）

**注意**：必须在单独的终端中运行 API 服务器（`src/run.py --port 8001`），演示才能正常工作。

---

### 方式 B：完整功能模式

#### 步骤 1：提取记忆

运行提取脚本从对话数据中提取记忆：

```bash
# 启动 API 服务器（如果还没运行）
uv run python src/bootstrap.py src/run.py --port 8001

# 在另一个终端运行提取脚本
uv run python src/bootstrap.py demo/extract_memory.py
```

该脚本将：
- 清空所有已存在的记忆数据
- 加载 `data/assistant_chat_zh.json` 对话文件
- 逐条发送到 API 服务器进行记忆提取
- 所有记忆存储到数据库（MongoDB、Elasticsearch、Milvus）

> **💡 提示**：`extract_memory.py` 脚本简单明了，直接调用 HTTP API。您可以修改脚本中的 `data_file` 和 `profile_scene` 变量来使用不同的数据文件。

#### 步骤 2：启动对话

运行对话脚本开始与 AI 对话：

```bash
# 推荐：使用 uv（从项目根目录执行）
uv run python src/bootstrap.py demo/chat_with_memory.py

# 备选：直接执行（从 demo 目录执行）
cd demo
python chat_with_memory.py
```

**交互选择**：
1. **语言选择**：选择 `[1] 中文` 或 `[2] English`
2. **场景选择**：选择 `[1] 助手模式` 或 `[2] 群聊模式`
3. **群组选择**：从 MongoDB 中加载的可用群组中选择
4. **检索模式**：选择 RRF（推荐）、Embedding、BM25 或 Agentic

**对话功能**：
- 💬 自然语言对话，AI 基于记忆上下文回答
- 🔍 自动检索相关记忆（显示检索结果）
- 📝 对话历史自动保存到 `chat_history/` 目录
- 🧠 输入特殊命令查看详细信息（`help`、`clear`、`reload`、`exit`）

---

## 📁 数据文件说明

系统使用 `data/` 目录中的示例对话文件：

| 场景 | 语言 | 文件名 |
|-----|------|--------|
| 助手对话 | 中文 | `data/assistant_chat_zh.json` |
| 助手对话 | 英文 | `data/assistant_chat_en.json` |
| 群组对话 | 中文 | `data/group_chat_zh.json` |
| 群组对话 | 英文 | `data/group_chat_en.json` |

所有数据文件遵循 [GroupChatFormat](../data_format/group_chat/group_chat_format.md) 规范。详见[数据说明文档](../data/README_zh.md)。

**使用自定义数据**：
编辑 `extract_memory.py`，修改 `data_file` 和 `profile_scene` 变量指向您的数据文件。

## 💬 对话命令

在对话会话期间，支持以下命令：

- **正常输入**：直接输入问题，AI 会基于记忆回答
- `help` - 显示帮助信息
- `clear` - 清空当前对话历史
- `reload` - 重新加载记忆和画像
- `exit` - 保存对话历史并退出
- `Ctrl+C` - 中断并保存

## ⚙️ 环境配置

在项目根目录创建 `.env` 文件（参考 `env.template`）：

```bash
# LLM 配置
LLM_MODEL=your_model
LLM_API_KEY=your_api_key
LLM_BASE_URL=your_base_url

# 嵌入模型配置
EMB_BASE_URL=http://localhost:11000/v1/embeddings
EMB_MODEL=Qwen3-Embedding-4B

# MongoDB 配置
MONGODB_URI=mongodb://admin:memsys123@localhost:27017
```

## 🔗 相关文档

- [群聊格式规范](../data_format/group_chat/group_chat_format.md)
- [API 文档](../docs/api_docs/agentic_v3_api_zh.md)
- [数据说明文档](../data/README_zh.md)
- [国际化使用指南](../docs/dev_docs/chat_i18n_usage.md)

## 📖 演示数据说明

### 群聊场景 (group_chat_en.json / group_chat_zh.json)

**项目背景：** AI 产品工作群，记录团队开发"智能销售助手"的完整历程

**核心内容：**
- MVP 开发阶段：基于 RAG 的问答系统
- 高级功能迭代：情绪识别、记忆系统
- 团队协作实践：从需求到交付的完整流程

**可用语言：** 英文和中文版本

**适合探索：** 团队协作模式、项目管理、技术方案演进

### 助手场景 (assistant_chat_en.json / assistant_chat_zh.json)

**对话背景：** 个人健康与生活助手，记录近 2 个月的连续交互

**核心内容：**
- 旅行规划：美食推荐、行程建议
- 健康管理：体重监测、饮食指导
- 运动康复：训练建议、伤后恢复

**可用语言：** 英文和中文版本

**适合探索：** 个性化服务、长期记忆积累、上下文理解

## ❓ 推荐问题示例

**群聊 AI 场景问题推荐：**
- Alex/Betty/... 在情绪识别项目中做了什么工作？
- 从情绪识别项目中，可以看出 Alex/Betty/... 具备什么样的工作能力？
- 情绪识别项目的交付结果如何？
- 记忆系统项目的进展如何？

**助手 AI 场景问题推荐：**
- 请为我推荐适合我的运动。
- 请为我推荐我可能喜欢的食物。
- 我的健康状况如何？

## 🔗 相关文档

- 📋 [群聊格式规范](../data_format/group_chat/group_chat_format.md) - 数据文件格式说明
- 🔌 [API 文档](../docs/api_docs/agentic_v3_api_zh.md) - API 接口文档
- 📦 [数据说明](../data/README_zh.md) - 示例数据详细说明
- 🏠 [项目主页](../README_zh.md) - 项目概述和架构
- 📘 [批量记忆化使用指南](../docs/dev_docs/run_memorize_usage.md) - 高级用法

## ❓ 常见问题

### Q: 找不到 API 服务器连接？
**A**: 确保先启动 API 服务器：`uv run python src/bootstrap.py src/run.py --port 8001`

### Q: extract_memory.py 如何使用自定义数据？
**A**: 编辑脚本，修改以下变量：
- `data_file`: 指向您的 JSON 数据文件
- `profile_scene`: 设置为 `"assistant"` 或 `"group_chat"`
- `base_url`: API 服务器地址（默认 `http://localhost:8001`）

### Q: 数据存储在哪里？
**A**: 所有记忆数据通过 HTTP API 存储到数据库：
- **MongoDB**: 存储 MemCell、情节、画像
- **Elasticsearch**: 关键词索引（BM25）
- **Milvus**: 向量索引（语义检索）
- **本地文件**: 仅 `chat_history/` 目录保存对话历史

### Q: 支持哪些场景？
**A**: 支持两种场景：
- **助手模式（assistant）**：一对一对话，提取个性化画像
- **群聊模式（group_chat）**：多人对话，提取群组记忆和成员画像

### Q: 数据文件格式是什么？
**A**: JSON 格式，遵循 [GroupChatFormat](../data_format/group_chat/group_chat_format.md) 规范。项目提供 4 个示例文件供参考。

### Q: 如何查看数据库中的数据？
**A**: 
- **MongoDB**: 使用 MongoDB Compass 或命令行查询
- **检索测试**: 运行 `demo/tools/test_retrieval_comprehensive.py`
- **清空数据**: 运行 `demo/tools/clear_all_data.py`

## 💡 需要帮助？

- 🏠 查看主 [README](../README_zh.md) 了解项目设置和架构
- 💬 在 GitHub 上提交问题
- 📧 联系项目维护者

---

**祝您探索愉快！🧠✨**

