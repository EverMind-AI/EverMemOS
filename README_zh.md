<div align="center">

<h1>
  <img src="figs/logo.png" alt="EverMemOS Logo" height="40" style="vertical-align: middle; margin-right: 12px;"/>
  EverMemOS
</h1>

<p><strong>每次交流，都由理解驱动</strong> · 企业级智能记忆系统</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-0084FF?style=flat-square&logo=python&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-00B894?style=flat-square&logo=apache&logoColor=white" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Supported-4A90E2?style=flat-square&logo=docker&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Latest-26A69A?style=flat-square&logo=fastapi&logoColor=white" />
  <img alt="MongoDB" src="https://img.shields.io/badge/MongoDB-7.0+-00C853?style=flat-square&logo=mongodb&logoColor=white" />
  <img alt="Elasticsearch" src="https://img.shields.io/badge/Elasticsearch-8.x-0084FF?style=flat-square&logo=elasticsearch&logoColor=white" />
  <img alt="Milvus" src="https://img.shields.io/badge/Milvus-2.4+-00A3E0?style=flat-square" />
  <img alt="Redis" src="https://img.shields.io/badge/Redis-7.x-26A69A?style=flat-square&logo=redis&logoColor=white" />
   <a href="https://github.com/EverMind-AI/EverMemOS/releases">
    <img alt="Release" src="https://img.shields.io/badge/release-v1.0.0-4A90E2?style=flat-square" />
  </a>
</p>

<p>
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

</div>

---

> 💬 **不止记忆，更是远见。**

**EverMemOS** 是一个着眼未来的**智能系统**。  
传统的 AI 记忆仅是“回顾过去”的数据库，而 EverMemOS 让 AI 不仅能“记住”发生了什么，更能“理解”这些记忆的意义，并据此指导当下的行动与决策。在EverMemOS的演示工具中，你可以看到EverMemOS如何从你的历史信息中提取重要信息，然后在对话时记住你的**喜好、习惯和历史**，就像一个真正认识你的**朋友**。
在 **LoCoMo** 基准测试中，我们基于 EverMemOS 的方法在 **LLM-Judge** 评测下达到了  **92.3% 的推理准确率**，优于我们测试的同类方法。

---

## 📢 最新动态

<table>
<tr>
<td width="100%" style="border: none;">

**[2025-11-02] 🎉 🎉 🎉 EverMemOS v1.0.0 版本发布！**

- ✨  **稳定版本**：AI 记忆系统正式开源  
- 📚  **文档完善**：提供快速开始指南与完整 API 说明 
- 📈  **基准测试**：LoCoMo数据集基准测试流程
- 🖥️  **演示工具**：用容易上手的demo快速开始

</td>
</tr>
</table>

---

## 🎯 核心愿景  
构建永不遗忘的 AI 记忆，让每一次对话都建立在前序理解之上。

---

## 💡 独特优势

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>🔗 脉络有绪</h3>
      <p><strong>不止“碎片”，串联“故事”</strong>：自动串联对话片段，构建清晰主题脉络，让 AI “看得明白”。</p>
      <blockquote>
        面对多线程对话时，它能自然地区分“A 项目的进度讨论”和“B 团队的策略规划”，并在每个主题中维持连贯的上下文逻辑。<br/><br/>
        从零散片语到完整叙事，AI 不再“听懂一句”，而是“听懂整件事”。
      </blockquote>
    </td>
    <td width="33%" valign="top">
      <h3>🧠 感知有据</h3>
      <p><strong>不止“检索”，智能“感知”</strong>：主动捕捉记忆与任务间的深层关联，让 AI 在关键时刻“想得周到”。</p>
      <blockquote>
        想象一下：当用户请求“推荐食物”时，AI 会主动联想到“你两天前刚做了牙科手术”这一关键信息，自动调整建议，避开不适宜的选项。<br/><br/>
        这是一种 <strong>上下文自觉 (Contextual Awareness)</strong> —— 让 AI 的思考真正建立在理解之上，而非孤立回应。
      </blockquote>
    </td>
    <td width="33%" valign="top">
      <h3>💾 画像有灵</h3>
      <p><strong>不止“档案”，动态“成长”</strong>：实时更新用户画像，越聊越懂你，让 AI “认得真切”。</p>
      <blockquote>
        你的每一次交流都会悄然更新 AI 对你的理解——偏好、风格、关注点都在持续演化。<br/><br/>
        随着互动的深入，它不只是“记住你说过什么”，而是在“学习你是谁”。
      </blockquote>
    </td>
  </tr>
</table>

---

## 📑 目录


<div align="center">
<table>
<tr>
<td width="50%" valign="top">

- [📖 项目介绍](#-项目介绍)
- [🎯 系统框架](#-系统框架)
- [📁 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
  - [如何使用](#如何使用)
  - [更多详细信息](#更多详细信息)

</td>
<td width="50%" valign="top">

- [📚 文档](#-文档)
  - [开发文档](#开发文档)
  - [API 文档](#api-文档)
  - [核心框架](#核心框架)
- [🏗️ 架构设计](#️-架构设计)
- [🤝 贡献](#-贡献)
- [🌟 加入我们](#-加入我们)
- [🙏 致谢](#-致谢)

</td>
</tr>
</table>
</div>

---

## 📖 项目介绍

**EverMemOS** 是一个开源项目，旨在为对话式 AI 智能体提供长期记忆能力。它从对话中提取、构建和检索信息，使智能体能够维持上下文、回忆过去的互动，并逐步建立用户画像。这使得对话变得更具个性化、连贯性和智能。

> 📄 **论文即将发布** - 我们的技术论文正在准备中，敬请期待！

## 🎯 系统框架

EverMemOS 围绕两条主线运行：**记忆构筑**与**记忆感知**。它们组成认知闭环，使系统持续吸收、沉淀并运用过往信息，让每次回应立足真实上下文与长期记忆。

<p align="center">
  <img src="figs/overview.png" alt="Overview" />
</p>

### 🧩 记忆构筑

记忆构筑层：根据原始对话数据构筑结构化、可检索的长期记忆。

- **核心要素**
  - ⚛️ **记忆单元**：从对话中提炼的核心记忆结构单元，便于后续组织与引用
  - 🗂️ **多层次记忆**：将相关片段按主题与脉络整合，形成可复用的多层次记忆
  - 🏷️ **多类型记忆**：覆盖情节、画像、偏好、关系、语义知识、基础事实与核心记忆

- **工作流程**
  1. **记忆单元提取**：识别对话中的关键信息，生成记忆单元
  2. **结构记忆构筑**：按主题与参与者整合，形成情节与画像
  3. **智能存储索引**：持久化保存，并建立关键词与语义索引，支持快速召回

### 🔎 记忆感知

记忆感知层：针对查询快速召回相关记忆，通过多轮推理与智能融合，实现精准的上下文感知。

#### 🎯 智能检索工具

- **🧪 混合检索 (RRF 融合)**  
  并行执行语义与关键词检索，采用 Reciprocal Rank Fusion 算法无缝融合

- **📊 智能重排序 (Reranker)**  
  批量并发处理 + 指数退避重试，在高吞吐下保持稳定性  
  对候选记忆按深度相关性重新排序，让最关键的信息优先呈现

#### 🤖 Agentic 智能检索

- **🎓 LLM 引导的多轮召回**  
  对于不充分的情况，生成 2-3 个互补查询，并行检索并融合
  自动识别缺失信息，主动补足检索盲区

- **🔀 多查询并行策略**  
  当单一查询无法完整表达意图时，生成多个互补视角的查询  
  通过多路 RRF 融合，提升复杂意图的理解覆盖度

- **⚡ 轻量级快速模式**  
  对延迟敏感的场景，跳过 LLM 调用，RRF融合混合检索
  在速度与质量间灵活平衡

#### 🧠 推理融合

- **上下文整合**：将召回的多层次记忆（情节、画像、偏好）与当前对话拼接
- **可追溯推理**：模型基于明确的记忆证据生成回复，避免幻觉

💡 通过 **"结构化记忆 → 多策略召回 → 智能检索 → 上下文推理"** 的认知闭环，让 AI 始终"带着记忆思考"，实现真正的上下文自觉。


## 📁 项目结构

<details>
<summary>展开/收起 目录结构</summary>

```
memsys-opensource/
├── src/                              # 源代码目录
│   ├── agentic_layer/                # 代理层 - 统一记忆接口
│   ├── memory_layer/                 # 记忆层 - 记忆提取
│   │   ├── memcell_extractor/        # MemCell提取器
│   │   ├── memory_extractor/         # Memory提取器
│   │   └── prompts/                  # LLM提示词模板
│   ├── retrieval_layer/              # 检索层 - 记忆检索
│   ├── biz_layer/                    # 业务层 - 业务逻辑
│   ├── infra_layer/                  # 基础设施层
│   ├── core/                         # 核心功能(DI/生命周期/中间件)
│   ├── component/                    # 组件(LLM适配器等)
│   └── common_utils/                 # 通用工具
├── demo/                             # 演示代码
├── data/                             # 示例对话数据
├── evaluation/                       # 评估脚本
│   └── src/                          # 评估框架源代码
├── data_format/                      # 数据格式定义
├── docs/                             # 文档
├── config.json                       # 配置文件
├── env.template                      # 环境变量模板
├── pyproject.toml                    # 项目配置
└── README.md                         # 项目说明
```

</details>

## 🚀 快速开始

### 环境要求

- Python 3.10+
- uv
- Docker 20.10+ 和 Docker Compose 2.0+
- **至少 4GB 可用内存**（用于 Elasticsearch 和 Milvus）

### 安装步骤

#### 使用 Docker 启动依赖服务 ⭐

使用 Docker Compose 一键启动所有依赖服务（MongoDB、Elasticsearch、Milvus、Redis）：

```bash
# 1. 克隆项目
git clone https://github.com/EverMind-AI/EverMemOS.git
cd EverMemOS

# 2. 启动 Docker 服务
docker-compose up -d

# 3. 验证服务状态
docker-compose ps

# 4. 安装 uv（如果还没有安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 5. 安装项目依赖
uv sync

# 6. 配置环境变量
cp env.template .env
# 编辑 .env 文件，填入必要的配置
#   - LLM_API_KEY: 填入您的 LLM API Key（用于记忆提取）
#   - DEEPINFRA_API_KEY: 填入您的 DeepInfra API Key（用于 Embedding 和 Rerank）
```

**Docker 服务说明**：
| 服务 | 宿主机端口 | 容器端口 | 用途 |
|------|-----------|---------|------|
| **MongoDB** | 27017 | 27017 | 主数据库，存储记忆单元和画像 |
| **Elasticsearch** | 19200 | 9200 | 关键词检索引擎（BM25） |
| **Milvus** | 19530 | 19530 | 向量数据库，语义检索 |
| **Redis** | 6379 | 6379 | 缓存服务 |

> 💡 **连接提示**：
> - 连接时使用**宿主机端口**（如 `localhost:19200` 访问 Elasticsearch）
> - MongoDB 凭据：`admin` / `memsys123`（仅用于本地开发）
> - 停止服务：`docker-compose down` | 查看日志：`docker-compose logs -f`

> 📖 MongoDB 详细安装指南：[MongoDB Installation Guide](docs/usage/MONGODB_GUIDE_zh.md)

---

### 如何使用

#### 🎯 运行演示：记忆提取和交互式聊天

演示部分展示了 EverMemOS 的端到端功能。

---

**🚀 快速开始：简单演示（推荐）** ⭐

体验 EverMemOS 最快的方式！只需 2 步就能看到记忆存储和检索的完整流程：

```bash
# 步骤 1：启动 API 服务器（终端 1）
uv run python src/bootstrap.py src/run.py --port 8001

# 步骤 2：运行简单演示（终端 2）
uv run python src/bootstrap.py demo/simple_demo.py
```

**它会做什么：**
- 存储 4 条关于运动爱好的对话消息
- 等待 10 秒建立索引
- 用 3 个不同的查询搜索相关记忆
- 展示完整的工作流程和友好的说明

**适合人群：** 首次使用者、快速测试、理解核心概念

查看演示代码 [`demo/simple_demo.py`](demo/simple_demo.py)

---

我们还设置了完整的体验场景：

**前置条件：启动 API 服务器**

```bash
# 终端 1：启动 API 服务器（必需）
uv run python src/bootstrap.py src/run.py --port 8001
```

> 💡 **提示**：API 服务器需要一直运行，请保持此终端打开。下面的所有操作都需要在另一个终端中进行。

---

**步骤 1: 提取记忆**

运行记忆提取脚本，处理示例对话数据并构建记忆数据库：

```bash
# 终端 2：运行提取脚本
uv run python src/bootstrap.py demo/extract_memory.py
```

该脚本将：
- 从 `data/` 目录读取对话数据
- 提取记忆单元并保存到配置的数据库中（例如 MongoDB）
- 生成用户画像并保存到 `demo/memcell_outputs/` 目录

> **💡 提示**:
> 配置非常简单！`extract_memory.py` 使用 HTTP API，只需 2 个参数即可运行。
> 详细的配置说明和使用指南请参阅 [Demo 文档](demo/README_zh.md)。

**步骤 2: 与记忆聊天**

提取记忆后，启动交互式聊天演示：

```bash
# 终端 2：运行聊天程序（确保 API 服务器仍在运行）
uv run python src/bootstrap.py demo/chat_with_memory.py
```

这将启动一个命令行界面，您可以与一个利用了刚提取的记忆的智能体进行对话。更多关于聊天功能的使用技巧和推荐问题，请参阅 [Demo 指南](demo/README_zh.md)。

**交互流程：**
1. **选择语言**：在中文和 English 之间选择界面语言。
2. **选择场景模式**：
   - **助手模式**：单人对话，基于个人记忆的智能助手。
   - **群聊模式**：多人群聊，基于群组记忆的对话分析。
3. **选择对话群组**：从数据库中可用的群组中选择。
4. **开始聊天**：与记忆增强的 AI 智能体互动。

---

#### 📊 运行评估：性能测试

评估框架提供了一种统一的模块化方法来对标准数据集（LoCoMo、LongMemEval、PersonaMem）上的记忆系统进行基准测试。

**快速测试（冒烟测试）**：

```bash
# 使用有限数据测试以验证一切正常
# 默认：第一个对话，前 10 条消息，前 3 个问题
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# 自定义冒烟测试：20 条消息，5 个问题
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --smoke-messages 20 --smoke-questions 5

# 测试不同数据集
uv run python -m evaluation.cli --dataset longmemeval --system evermemos --smoke
uv run python -m evaluation.cli --dataset personamem --system evermemos --smoke

# 测试特定阶段（例如只测试搜索和回答阶段）
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --stages search answer

# 快速查看冒烟测试结果
cat evaluation/results/locomo-evermemos-smoke/report.txt
```

**完整评估**：

```bash
# 在 LoCoMo 基准上评估 EvermemOS
uv run python -m evaluation.cli --dataset locomo --system evermemos

# 在其他数据集上评估
uv run python -m evaluation.cli --dataset longmemeval --system evermemos
uv run python -m evaluation.cli --dataset personamem --system evermemos

# 使用 --run-name 区分多次运行（用于 A/B 测试）
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name baseline
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name experiment1

# 如果中断则从检查点恢复（自动）
# 只需重新运行相同命令 - 它会检测并从检查点恢复
uv run python -m evaluation.cli --dataset locomo --system evermemos
```

**查看结果**：

```bash
# 结果保存到 evaluation/results/{dataset}-{system}[-{run-name}]/
cat evaluation/results/locomo-evermemos/report.txt          # 摘要指标
cat evaluation/results/locomo-evermemos/eval_results.json   # 每个问题的详细结果
cat evaluation/results/locomo-evermemos/pipeline.log        # 执行日志
```

评估流程包含 4 个阶段（添加 → 搜索 → 回答 → 评估），支持自动检查点和恢复。

> **⚙️ 评估配置**:
> - **数据准备**：需要将数据集放置在 `evaluation/data/` 中（参见 `evaluation/README.md`）
> - **环境配置**：在 `.env` 中配置 LLM API 密钥（参见 `env.template`）
> - **安装依赖**：运行 `uv sync --group evaluation` 安装依赖
> - **自定义配置**：复制并修改 `evaluation/config/systems/` 或 `evaluation/config/datasets/` 中的 YAML 文件
> - **高级用法**：参见 `evaluation/README.md` 了解检查点管理、特定阶段运行和系统对比

---

#### 🔌 调用 API 接口

**前置条件：启动 API 服务器**

在调用 API 之前，请确保已启动 API 服务器：

```bash
# 启动 API 服务器
uv run python src/bootstrap.py src/run.py --port 8001
```

> 💡 **提示**：API 服务器需要一直运行，请保持此终端打开。下面的 API 调用需要在另一个终端中进行。

---

使用 V3 API 存储单条消息记忆：

<details>
<summary>示例：存储单条消息</summary>

```bash
curl -X POST http://localhost:8001/api/v3/agentic/memorize \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+08:00",
    "sender": "user_103",
    "sender_name": "Chen",
    "content": "我们需要在本周完成产品设计",
    "group_id": "group_001",
    "group_name": "项目讨论组",
    "scene": "group_chat"
  }'
```

</details>
> ℹ️ `scene` 为必填字段，仅支持 `assistant` 或 `group_chat`，用于指定记忆提取策略。
**API 功能说明**：

- **`/api/v3/agentic/memorize`**: 存储单条消息记忆
- **`/api/v3/agentic/retrieve_lightweight`**: 轻量级记忆检索（Embedding + BM25 + RRF）
- **`/api/v3/agentic/retrieve_agentic`**: Agentic 记忆检索（LLM 引导的多轮智能检索）

更多 API 详情请参考 [Agentic V3 API 文档](docs/api_docs/agentic_v3_api_zh.md)。

---

**🔍 检索记忆**

EverMemOS 提供两种检索模式：**轻量级检索**（快速）和 **Agentic 检索**（智能）。

**轻量级检索**

| 参数 | 必填 | 说明 |
|------|------|------|
| `query` | 是* | 自然语言查询（*profile 数据源时可选） |
| `user_id` | 否 | 用户 ID |
| `data_source` | 是 | `episode` / `event_log` / `semantic_memory` / `profile` |
| `memory_scope` | 是 | `personal`（仅 user_id） / `group`（仅 group_id） / `all`（两者） |
| `retrieval_mode` | 是 | `embedding` / `bm25` / `rrf`（推荐） |
| `group_id` | 否 | 群组 ID |
| `current_time` | 否 | 过滤有效期内的 semantic_memory（格式: YYYY-MM-DD） |
| `top_k` | 否 | 返回结果数（默认: 5） |

**示例 1：个人记忆**

<details>
<summary>示例：个人记忆检索</summary>

```bash
curl -X POST http://localhost:8001/api/v3/agentic/retrieve_lightweight \
  -H "Content-Type: application/json" \
  -d '{
    "query": "用户喜欢什么运动",
    "user_id": "user_001",
    "data_source": "episode",
    "memory_scope": "personal",
    "retrieval_mode": "rrf"
  }'
```

</details>

**示例 2：群组记忆**

<details>
<summary>示例：群组记忆检索</summary>

```bash
curl -X POST http://localhost:8001/api/v3/agentic/retrieve_lightweight \
  -H "Content-Type: application/json" \
  -d '{
    "query": "讨论项目进展",
    "group_id": "project_team_001",
    "data_source": "episode",
    "memory_scope": "group",
    "retrieval_mode": "rrf"
  }'
```

</details>

---

**Agentic 检索**

使用 LLM 引导的多轮智能搜索，自动进行查询改进和结果重排序。

<details>
<summary>示例：Agentic 检索</summary>

```bash
curl -X POST http://localhost:8001/api/v3/agentic/retrieve_agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "用户可能喜欢吃什么？",
    "user_id": "user_001",
    "group_id": "chat_group_001",
    "top_k": 20,
    "llm_config": {
      "model": "gpt-4o-mini",
      "api_key": "your_api_key"
    }
  }'
```

</details>

> ⚠️ Agentic 检索需要 LLM API Key，耗时较长，但能为需要多记忆来源、复杂逻辑查询提供更高质量的结果。

> 📖 完整文档：[Agentic V3 API](docs/api_docs/agentic_v3_api_zh.md) | 测试工具：`demo/tools/test_retrieval_comprehensive.py`

---

#### 📦 批量存储群聊记忆

EverMemOS 支持标准化的群聊数据格式（[GroupChatFormat](data_format/group_chat/group_chat_format.md)），可以使用脚本批量存储：

```bash
# 使用脚本批量存储（中文数据）
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_zh.json \
  --api-url http://localhost:8001/api/v3/agentic/memorize \
  --scene group_chat

# 或者使用英文数据
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_en.json \
  --api-url http://localhost:8001/api/v3/agentic/memorize \
  --scene group_chat

# 验证文件格式
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_zh.json \
  --scene group_chat \
  --validate-only
```

> ℹ️ **Scene 参数说明**：`scene` 为必填字段，用于指定记忆提取策略：
> - 使用 `assistant` 用于一对一助手对话
> - 使用 `group_chat` 用于多人群聊讨论
> 
> **注意**：在数据文件中，您可能会看到 `scene` 值为 `work` 或 `company` - 这些是数据格式中的内部场景描述符。命令行参数 `--scene` 使用不同的值（`assistant`/`group_chat`）来指定应用哪个提取流水线。

**GroupChatFormat 格式示例**：

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "group_id": "group_001",
    "name": "项目讨论组",
    "user_details": {
      "user_101": {
        "full_name": "Alice",
        "role": "产品经理"
      }
    }
  },
  "conversation_list": [
    {
      "message_id": "msg_001",
      "create_time": "2025-02-01T10:00:00+08:00",
      "sender": "user_101",
      "content": "大家早上好"
    }
  ]
}
```

完整的格式说明请参考 [群聊格式规范](data_format/group_chat/group_chat_format.md)。

### 更多详细信息

详细的安装、配置和使用说明，请参考：
- 📚 [快速开始指南](docs/dev_docs/getting_started.md) - 完整的安装和配置步骤
- 📖 [API 使用指南](docs/dev_docs/api_usage_guide.md) - API 接口和数据格式详解
- 🔧 [开发指南](docs/dev_docs/development_guide.md) - 架构设计和开发最佳实践
- 🚀 [Bootstrap 使用](docs/dev_docs/bootstrap_usage.md) - 脚本运行器使用说明
- 📝 [群聊格式规范](data_format/group_chat/group_chat_format.md) - 标准化数据格式


## 📚 文档

### 开发文档
- [快速开始指南](docs/dev_docs/getting_started.md) - 安装、配置和启动
- [开发指南](docs/dev_docs/development_guide.md) - 架构设计和最佳实践
- [Bootstrap 使用](docs/dev_docs/bootstrap_usage.md) - 脚本运行器

### API 文档
- [Agentic V3 API](docs/api_docs/agentic_v3_api_zh.md) - 智能体层 API

### 核心框架
- [依赖注入框架](src/core/di/README.md) - DI 容器使用指南

### 演示与评估
- [📖 演示指南](demo/README_zh.md) - 交互式示例和记忆提取演示
- [📊 数据指南](data/README_zh.md) - 示例对话数据和格式规范
- [📊 评估指南](evaluation/README_zh.md) - 在标准基准测试上测试基于EverMemOS的方法

## 🏗️ 架构设计

EverMemOS 采用分层架构设计，主要包括：

- **智能体层（Agentic Layer）**: 记忆提取、向量化、检索和重排序
- **记忆层（Memory Layer）**: 记忆单元提取、情景记忆管理
- **检索层（Retrieval Layer）**: 多模态检索和结果排序
- **业务层（Biz Layer）**: 业务逻辑和数据操作
- **基础设施层（Infra Layer）**: 数据库、缓存、消息队列等适配器
- **核心框架（Core）**: 依赖注入、中间件、队列管理等

更多架构细节请参考[开发指南](docs/dev_docs/development_guide.md)。

## 🤝 贡献

我们欢迎所有形式的贡献！无论是报告 Bug、提出新功能建议，还是提交代码改进，都非常感谢。

在开始之前，请阅读我们的 [贡献指南](CONTRIBUTING.md)，快速了解开发环境、代码规范、Git 提交流程与 Pull Request 要求。

## 🌟 加入我们

<!-- 
此部分可以添加：
- 社区交流方式（Discord、Slack、微信群等）
- 技术讨论论坛
- 定期会议信息
- 联系邮箱
-->

我们正在构建一个充满活力的开源社区！

### 联系方式

<p>
  <a href="https://github.com/EverMind-AI/EverMemOS/issues"><img alt="GitHub Issues" src="https://img.shields.io/badge/GitHub-Issues-blue?style=flat-square&logo=github"></a>
  <a href="https://github.com/EverMind-AI/EverMemOS/discussions"><img alt="GitHub Discussions" src="https://img.shields.io/badge/GitHub-Discussions-blue?style=flat-square&logo=github"></a>
  <a href="mailto:evermind@shanda.com"><img alt="Email" src="https://img.shields.io/badge/Email-联系我们-blue?style=flat-square&logo=gmail"></a>
  <a href="https://www.reddit.com/r/EverMindAI/"><img alt="Reddit" src="https://img.shields.io/badge/Reddit-r/EverMindAI-orange?style=flat-square&logo=reddit"></a>
  <a href="https://x.com/EverMindAI"><img alt="X" src="https://img.shields.io/badge/X-@EverMindAI-black?style=flat-square&logo=x"></a>
</p>

### 贡献者

感谢所有为这个项目做出贡献的开发者！

<!-- 可以使用 GitHub Contributors 自动生成 -->
<!-- <a href="https://github.com/your-org/memsys_opensource/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=your-org/memsys_opensource" />
</a> -->

## 📖 引用

如果您在研究中使用了 EverMemOS，请引用我们的论文（即将发布）：

```
Coming soon
```

## 📄 许可证

本项目采用 [Apache 2.0 许可证](LICENSE)。这意味着你可以自由地使用、修改和分发本项目，但需要遵守以下主要条件：
- 必须包含 Apache 2.0 许可证副本
- 必须声明对代码所做的重大修改
- 必须保留所有版权、专利、商标和归属声明
- 如果包含 NOTICE 文件，必须在分发时包含该文件

## 🙏 致谢

<!-- 
此部分可以添加：
- 受启发的项目
- 使用的开源库
- 支持的组织或个人
-->

感谢以下项目和社区的灵感和支持：

- [Memos](https://github.com/usememos/memos) - 感谢 Memos 项目提供了一个完善的、标准化的开源笔记服务，为我们的记忆系统设计提供了宝贵的启发。

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️**

Made with ❤️ by the EverMemOS Team

</div>