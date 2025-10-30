<div align="center">

<h1>EverMemOS 🧠</h1>

<p><strong>每次交流，都由理解驱动</strong> · 企业级智能记忆系统</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green.svg" />
  <img alt="Status" src="https://img.shields.io/badge/Status-Production-brightgreen.svg" />
</p>

<p>
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

</div>

---

> 💬 **不止记忆，更是远见。**

**EverMemOS** 是一个着眼未来的智能系统。  
传统的 AI 记忆仅是“回顾过去”的数据库，而 EverMemOS 让 AI 不仅能“记住”发生了什么，更能“理解”这些记忆的意义，并据此指导当下的行动与决策。在 **LoCoMo** 基准测试中，我们基于 EverMemOS 的方法在 **LLM-Judge** 评测下达到了  **92.3% 的推理准确率**，优于我们测试的同类方法。

---

## 📢 最新动态

<table>
<tr>
<td width="100%" style="border: none;">

**[2025-10-28] 🎉 v1.0.0 稳定版本发布！**

- ✨ **稳定版本**：AI 记忆系统正式开源  
- 📚 **文档完善**：提供快速开始指南与完整 API 说明  

</td>
</tr>
</table>

---

## 🎯 核心愿景  
构建永不遗忘的 AI 记忆，让每一次对话都建立在前序理解之上。

---

## 💡 独特优势

### 🔗 脉络有绪  
不止“碎片”，串联“故事”：自动串联对话片段，构建清晰主题脉络，让 AI “看得明白”。  

> 面对多线程对话时，它能自然地区分“A 项目的进度讨论”和“B 团队的策略规划”，并在每个主题中维持连贯的上下文逻辑。  
> 从零散片语到完整叙事，AI 不再“听懂一句”，而是“听懂整件事”。

---

### 🧠 感知有据  
不止“检索”，智能“感知”：主动捕捉记忆与任务间的深层关联，让 AI 在关键时刻“想得周到”。  

> 想象一下：当用户请求“推荐食物”时，AI 会主动联想到“你两天前刚做了牙科手术”这一关键信息，自动调整建议，避开不适宜的选项。  
> 这是一种 **上下文自觉 (Contextual Awareness)** —— 让 AI 的思考真正建立在理解之上，而非孤立回应。

---

### 💾 画像有灵  
不止“档案”，动态“成长”：实时更新用户画像，越聊越懂你，让 AI “认得真切”。  

> 你的每一次交流都会悄然更新 AI 对你的理解——偏好、风格、关注点都在持续演化。  
> 随着互动的深入，它不只是“记住你说过什么”，而是在“学习你是谁”。

---

## 📑 目录


<div align="center">
<table>
<tr>
<td width="50%" valign="top">

- [📖 项目介绍](#-项目介绍)
- [🎯 核心概念](#-核心概念)
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

**EverMemOS** 是一个开源项目，旨在为对话式 AI 智能体提供长期记忆能力。本代码库是论文《EverMemOS》的官方实现。它从对话中提取、构建和检索信息，使智能体能够维持上下文、回忆过去的互动，并逐步建立用户画像。这使得对话变得更具个性化、连贯性和智能。

## 🎯 系统框架

EverMemOS 围绕两条主线运行：**记忆构筑**与**记忆感知**。它们组成认知闭环，使系统持续吸收、沉淀并运用过往信息，让每次回应立足真实上下文与长期记忆。

<p align="center">
  <img src="figs/overview.png" alt="Overview" />
</p>

### 🧩 记忆构筑

记忆构筑层：根据原始对话数据构筑结构化、可检索的长期记忆。

- **核心要素**
  - ⚛️ **原子记忆单元 MemCell**：从对话中提炼的核心记忆结构单元，便于后续组织与引用
  - 🗂️ **多层次记忆**：将相关片段按主题与脉络整合，形成可复用的多层次记忆
  - 🏷️ **多类型记忆**：覆盖情节、画像、偏好、关系、语义知识、基础事实与核心记忆

- **工作流程**
  1. **MemCell 提取**：识别对话中的关键信息，生成原子记忆
  2. **记忆构筑**：按主题与参与者整合，形成情节与画像
  3. **存储与索引**：持久化保存，并建立关键词与语义索引，支持快速召回

### 🔎 记忆感知

记忆感知层：针对查询快速召回相关记忆，根据结果进行推理。

- **检索策略**
  - 🔎 **关键词检索**：适合明确术语与短句的精确匹配
  - 🧭 **语义检索**：面向含义相近的表达与跨表述召回
  - 🧪 **混合检索**：并行多路召回，融合去重，兼顾稳定性与覆盖度
  - 📊 **重排序**：轻量重排模型对候选按相关性重新排序，进一步提升结果质量

- **推理融合**
  - 🧠 将召回的记忆与当前上下文一并输入模型，生成更连贯、可追溯的响应

💡 通过"记忆构筑 → 检索/重排 → 推理融合"的闭环，让系统始终"带着记忆思考"。


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
│   └── locomo_evaluation/            # LoCoMo 基准测试
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
- uv（推荐的包管理器）
- [MongoDB Installation Guide](docs/usage/MONGODB_GUIDE_zh.md), Redis, Elasticsearch, Milvus (可选)

### 安装步骤


```bash
# 1. 克隆项目
git clone https://github.com/your-org/memsys_opensource.git
cd memsys_opensource

# 2. 安装 uv（如果还没有安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 安装项目依赖
uv sync

# 4. 配置环境变量
cp env.template .env

# 编辑 .env 文件，填入必要的配置
#   - LLM_API_KEY: 默认使用 OpenRouter，请填入您的 OpenRouter API Key。
#   - DEEPINFRA_API_KEY: 填入您的 DeepInfra API Key，用于 Embedding 和 Rerank 服务。
#   - 其他数据库 (MongoDB/Redis/ES/Milvus) 根据您的本地或远程部署情况进行配置。

```

### 如何使用

#### 1. 运行演示：记忆提取和交互式聊天

演示部分展示了 EverMemOS 的端到端功能。

**步骤 1: 提取记忆**

运行记忆提取脚本，处理示例对话数据并构建记忆数据库：

```bash
python demo/extract_memory.py
```

该脚本将：
- 从 `data/` 目录读取对话数据
- 提取记忆单元并保存到配置的数据库中（例如 MongoDB）
- 生成用户画像并保存到 `demo/memcell_outputs/` 目录

> **💡 提示**:
> 默认情况下，脚本会提取 **助手 (ASSISTANT)** 场景的记忆。您可以选择性地提取 **群聊 (GROUP_CHAT)** 场景的记忆：
> 1. 打开 `demo/extract_memory.py` 文件。
> 2. 找到 `EXTRACT_CONFIG` 配置部分。
> 3. 将 `scenario_type` 从 `ScenarioType.ASSISTANT` 修改为 `ScenarioType.GROUP_CHAT`。
> 4. 再次运行提取脚本。
>
> 您可以只运行一种场景，也可以两种都运行。

**步骤 2: 与记忆聊天**

提取记忆后，启动交互式聊天演示：

```bash
python demo/chat_with_memory.py
```

这将启动一个命令行界面，您可以与一个利用了刚提取的记忆的智能体进行对话。更多关于聊天功能的使用技巧和推荐问题，请参阅 [Demo 指南](demo/README_zh.md)。

**交互流程：**
1. **选择语言**：在中文和 English 之间选择界面语言。
2. **选择场景模式**：
   - **助手模式**：单人对话，基于个人记忆的智能助手。
   - **群聊模式**：多人群聊，基于群组记忆的对话分析。
3. **选择对话群组**：从数据库中可用的群组中选择。
4. **开始聊天**：与记忆增强的 AI 智能体互动。

**新功能：**
- ✨ **国际化 (i18n)**：完整支持中文和英文界面。
- 🎯 **运行时场景选择**：启动时可选择助手模式或群聊模式。
- 🌍 **增强用户体验**：所有 UI 元素、提示和错误消息都以您选择的语言显示。
- 📝 **智能上下文**：智能体会根据所选场景模式自动调整行为和提示词。

#### 2. 运行评估：性能测试

评估框架提供了一种系统化的方法来衡量记忆系统的性能，基于 LoCoMo 评估数据集。

```bash
# 阶段 1: 记忆单元提取
python evaluation/locomo_evaluation/stage1_memcells_extraction.py

# 阶段 2: 索引构建
python evaluation/locomo_evaluation/stage2_index_building.py

# 阶段 3: 记忆检索
python evaluation/locomo_evaluation/stage3_memory_retrivel.py

# 阶段 4: 回应生成
python evaluation/locomo_evaluation/stage4_response.py

# 阶段 5: 评估
python evaluation/locomo_evaluation/stage5_eval.py
```

每个脚本对应评估流水线中的一个阶段，从数据处理到性能评分。

> **⚙️ 评估配置**:
> 在运行评估前，您可以修改 `evaluation/locomo_evaluation/config.py` 文件来调整实验设置：
> - **`ExperimentConfig.experiment_name`**: 修改此变量可以更改实验结果的保存目录。
> - **`ExperimentConfig.llm_service`**: 选择要使用的 LLM 服务 (例如, `"openai"` 或 `"vllm"`)。
> - **`ExperimentConfig.llm_config`**: 在此字典中配置所选 LLM 服务的具体参数，如模型、API 地址 (`base_url`) 和密钥 (`api_key`)。

#### 3. 调用 API 接口

使用 V3 API 存储单条消息记忆：

```bash
curl -X POST http://localhost:1995/api/v3/agentic/memorize \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+08:00",
    "sender": "user_103",
    "sender_name": "Chen",
    "content": "我们需要在本周完成产品设计",
    "group_id": "group_001",
    "group_name": "项目讨论组"
  }'
```

#### 4. 批量存储群聊记忆

EverMemOS 支持标准化的群聊数据格式（[GroupChatFormat](data_format/group_chat/group_chat_format.md)），可以使用脚本批量存储：

```bash
# 使用脚本批量存储
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v3/agentic/memorize

# 验证文件格式
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --validate-only
```

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
- [依赖管理](docs/dev_docs/project_deps_manage.md) - 包管理和版本控制

### API 文档
- [Agentic V3 API](docs/api_docs/agentic_v3_api.md) - 智能体层 API
- [Agentic V2 API](docs/api_docs/agentic_v2_api.md) - 智能体层 API（旧版）

### 核心框架
- [依赖注入框架](src/core/di/README.md) - DI 容器使用指南

### 演示与评估
- [📖 演示指南](demo/README_zh.md) - 交互式示例和记忆提取演示
- [📊 数据指南](data/README_zh.md) - 示例对话数据和格式规范

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

我们欢迎所有形式的贡献！无论是报告 Bug、提出新功能建议，还是提交代码改进。

在开始贡献之前，请阅读我们的 [贡献指南](CONTRIBUTING.md)，了解：
- 开发环境设置
- 代码规范和最佳实践
- Git 提交规范（Gitemoji）
- Pull Request 流程

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

- **GitHub Issues**: [提交问题和建议](https://github.com/your-org/memsys_opensource/issues)
- **讨论区**: [参与讨论](https://github.com/your-org/memsys_opensource/discussions)
- **邮箱**: [待补充联系邮箱]
- **社区**: [待补充社区链接]

### 贡献者

感谢所有为这个项目做出贡献的开发者！

<!-- 可以使用 GitHub Contributors 自动生成 -->
<!-- <a href="https://github.com/your-org/memsys_opensource/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=your-org/memsys_opensource" />
</a> -->

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

- （待补充）

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐️**

Made with ❤️ by the EverMemOS Team

</div>