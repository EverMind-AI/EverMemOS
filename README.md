<div align="center">

<h1>EverMemOS 🧠</h1>

<p><strong>Let every interaction be driven by understanding.</strong> · Enterprise-Grade Intelligent Memory System</p>

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

> 💬 **More than memory — it's foresight.**

**EverMemOS** is a forward-thinking intelligent system.  
While traditional AI memory serves merely as a "look-back" database, EverMemOS enables AI not only to "remember" what happened, but also to "understand" the meaning behind these memories and use them to guide current actions and decisions. On the **LoCoMo** benchmark, our approach built upon EverMemOS achieved a reasoning accuracy of **92.3%** (evaluated by LLM-Judge), outperforming comparable methods in our evaluation.

---

## 📢 Latest Updates

<table>
<tr>
<td width="100%" style="border: none;">

**[2025-10-28] 🎉 v1.0.0 Stable Release!**

- ✨ **Stable Version**: AI Memory System officially open sourced  
- 📚 **Complete Documentation**: Quick start guide and comprehensive API documentation  

</td>
</tr>
</table>

---

## 🎯 Core Vision  
Build AI memory that never forgets, making every conversation built on previous understanding.

---

## 💡 Unique Advantages

### 🔗 Coherent Narrative  
Beyond "fragments," connecting "stories": Automatically linking conversation pieces to build clear thematic context, enabling AI to "truly understand."

> When facing multi-threaded conversations, it naturally distinguishes between "Project A progress discussion" and "Team B strategy planning," maintaining coherent contextual logic within each theme.  
> From scattered phrases to complete narratives, AI no longer just "understands one sentence" but "understands the whole story."

---

### 🧠 Evidence-Based Perception  
Beyond "retrieval," intelligent "perception": Proactively capturing deep connections between memories and tasks, enabling AI to "think thoroughly" at critical moments.

> Imagine: When a user asks for "food recommendations," the AI proactively recalls "you had dental surgery two days ago" as a key piece of information, automatically adjusting suggestions to avoid unsuitable options.  
> This is **Contextual Awareness** — enabling AI thinking to be truly built on understanding rather than isolated responses.

---

### 💾 Living Profiles  
Beyond "records," dynamic "growth": Real-time user profile updates that get to know you better with each conversation, enabling AI to "recognize you authentically."

> Every interaction subtly updates the AI's understanding of you — preferences, style, and focus points all continuously evolve.  
> As interactions deepen, it doesn't just "remember what you said," but is "learning who you are."

---

<h2>📑 Table of Contents</h2>


<div align="center">
<table>
<tr>
<td width="50%" valign="top">

 - [📖 Project Introduction](#-project-introduction)
 - [🎯 System Framework](#-system-framework)
 - [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [How to Use](#how-to-use)
  - [More Details](#more-details)

</td>
<td width="50%" valign="top">

- [📚 Documentation](#-documentation)
  - [Developer Docs](#developer-docs)
  - [API Documentation](#api-documentation)
  - [Core Framework](#core-framework)
- [🏗️ Architecture Design](#️-architecture-design)
- [🤝 Contributing](#-contributing)
- [🌟 Join Us](#-join-us)
- [🙏 Acknowledgments](#-acknowledgments)

</td>
</tr>
</table>
</div>

---

## 📖 Project Introduction

**EverMemOS** is an open-source project designed to provide long-term memory capabilities to conversational AI agents. This codebase is the official implementation of the paper "EverMemOS". It extracts, structures, and retrieves information from conversations, enabling agents to maintain context, recall past interactions, and progressively build user profiles. This results in more personalized, coherent, and intelligent conversations.

## 🎯 System Framework

EverMemOS operates along two main tracks: **memory construction** and **memory perception**. Together they form a cognitive loop that continuously absorbs, consolidates, and applies past information, so every response is grounded in real context and long-term memory.

<p align="center">
  <img src="figs/overview.png" alt="Overview" />
</p>

### 🧩 Memory Construction

Memory construction layer: builds structured, retrievable long-term memory from raw conversation data.

- **Core elements**
  - ⚛️ **Atomic memory unit MemCell**: the core structured unit distilled from conversations for downstream organization and reference
  - 🗂️ **Multi-level memory**: integrate related fragments by theme and storyline to form reusable, hierarchical memories
  - 🏷️ **Multiple memory types**: covering episodes, profiles, preferences, relationships, semantic knowledge, basic facts, and core memories

- **Workflow**
  1. **MemCell extraction**: identify key information in conversations to generate atomic memories
  2. **Memory construction**: integrate by theme and participants to form episodes and profiles
  3. **Storage and indexing**: persist data and build keyword and semantic indexes to support fast recall

### 🔎 Memory Perception

Memory perception layer: quickly recalls relevant memories for a query and reasons over them.

- **Retrieval strategies**
  - 🔎 **Keyword retrieval**: exact matching for explicit terms and short phrases
  - 🧭 **Semantic retrieval**: recall paraphrases and semantically similar expressions
  - 🧪 **Hybrid retrieval**: parallel multi-channel recall with fusion and deduplication for both stability and coverage
  - 📊 **Reranking**: a lightweight model reorders candidates by relevance to further improve the top results

- **Reasoning fusion**
  - 🧠 Feed the recalled memories together with the current context into the model to produce more coherent and traceable responses

💡 Through the loop of "Memory Construction → Retrieval/Reranking → Reasoning Fusion", the system always "thinks with memory".

## 📁 Project Structure

<details>
<summary>Expand/Collapse Directory Structure</summary>

```
memsys-opensource/
├── src/                              # Source code directory
│   ├── agentic_layer/                # Agentic layer - unified memory interface
│   ├── memory_layer/                 # Memory layer - memory extraction
│   │   ├── memcell_extractor/        # MemCell extractor
│   │   ├── memory_extractor/         # Memory extractor
│   │   └── prompts/                  # LLM prompt templates
│   ├── retrieval_layer/              # Retrieval layer - memory retrieval
│   ├── biz_layer/                    # Business layer - business logic
│   ├── infra_layer/                  # Infrastructure layer
│   ├── core/                         # Core functionality (DI/lifecycle/middleware)
│   ├── component/                    # Components (LLM adapters, etc.)
│   └── common_utils/                 # Common utilities
├── demo/                             # Demo code → [📖 Demo Guide](demo/README.md)
├── data/                             # Sample conversation data → [📊 Data Guide](data/README.md)
├── evaluation/                       # Evaluation scripts
├── data_format/                      # Data format definitions
├── docs/                             # Documentation
├── config.json                       # Configuration file
├── env.template                      # Environment variable template
├── pyproject.toml                    # Project configuration
└── README.md                         # Project description
```

</details>

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- uv (recommended package manager)
- [MongoDB Installation Guide](docs/usage/MONGODB_GUIDE.md), Redis, Elasticsearch, Milvus (optional)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/memsys_opensource.git
cd memsys_opensource

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install project dependencies
uv sync

# 4. Configure environment variables
cp env.template .env
# Edit the .env file and fill in the necessary configurations:
#   - LLM_API_KEY: Defaults to OpenRouter. Please enter your OpenRouter API Key.
#   - DEEPINFRA_API_KEY: Enter your DeepInfra API Key for Embedding and Rerank services.
#   - Other databases (MongoDB/Redis/ES/Milvus) should be configured according to your local or remote deployment.

```


### How to Use

#### 1. Run Demo: Memory Extraction and Interactive Chat

The demo showcases the end-to-end functionality of EverMemOS.

**Step 1: Extract Memories**

Run the memory extraction script to process sample conversation data and build the memory database:

```bash
python demo/extract_memory.py
```

This script will:
- Read conversation data from the `data/` directory
- Extract MemCells and save them to the configured database (e.g., MongoDB)
- Generate user profiles and save them to `demo/memcell_outputs/` directory

> **💡 Tip**:
> By default, the script extracts memories for the **ASSISTANT** scenario. You can optionally extract memories for the **GROUP_CHAT** scenario:
> 1. Open the `demo/extract_memory.py` file.
> 2. Locate the `EXTRACT_CONFIG` section.
> 3. Change `scenario_type` from `ScenarioType.ASSISTANT` to `ScenarioType.GROUP_CHAT`.
> 4. Run the extraction script again.
>
> You can run either one or both scenarios.

**Step 2: Chat with Memory**

After extracting memories, start the interactive chat demo:

```bash
python demo/chat_with_memory.py
```

This will launch a command-line interface where you can converse with an agent that utilizes the just-extracted memories. For more details on chat features, tips, and suggested questions, please see the [Demo Guide](demo/README.md).

**Interactive Workflow:**
1. **Select Language**: Choose between Chinese (中文) or English interface.
2. **Select Scenario Mode**:
   - **Assistant Mode**: One-on-one conversation with personal memory-based AI assistant.
   - **Group Chat Mode**: Multi-person chat with group memory-based conversation analysis.
3. **Select Conversation Group**: Choose from available groups in your database.
4. **Start Chatting**: Interact with the memory-enhanced AI agent.

**New Features:**
- ✨ **Internationalization (i18n)**: Full support for Chinese and English interfaces.
- 🎯 **Runtime Scenario Selection**: Choose between Assistant Mode and Group Chat Mode at startup.
- 🌍 **Enhanced User Experience**: All UI elements, prompts, and error messages are displayed in your selected language.
- 📝 **Smart Context**: The agent automatically adjusts its behavior and prompts based on the selected scenario mode.

#### 2. Run Evaluation: Performance Testing

The evaluation framework provides a systematic way to measure the performance of the memory system, based on the LoCoMo evaluation dataset.

```bash
# Stage 1: MemCell Extraction
python evaluation/locomo_evaluation/stage1_memcells_extraction.py

# Stage 2: Index Building
python evaluation/locomo_evaluation/stage2_index_building.py

# Stage 3: Memory Retrieval
python evaluation/locomo_evaluation/stage3_memory_retrivel.py

# Stage 4: Response Generation
python evaluation/locomo_evaluation/stage4_response.py

# Stage 5: Evaluation
python evaluation/locomo_evaluation/stage5_eval.py
```

Each script corresponds to a stage in the evaluation pipeline, from data processing to performance scoring.

> **⚙️ Evaluation Configuration**:
> Before running the evaluation, you can modify the `evaluation/locomo_evaluation/config.py` file to adjust the experiment settings:
> - **`ExperimentConfig.experiment_name`**: Change this to alter the save directory for the results.
> - **`ExperimentConfig.llm_service`**: Select the LLM service to use (e.g., `"openai"` or `"vllm"`).
> - **`ExperimentConfig.llm_config`**: Configure parameters for the selected LLM service in this dictionary, such as the model, `base_url`, and `api_key`.

#### 3. Call API Endpoints

Use V3 API to store single message memory:

```bash
curl -X POST http://localhost:1995/api/v3/agentic/memorize \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+08:00",
    "sender": "user_103",
    "sender_name": "Chen",
    "content": "We need to complete the product design this week",
    "group_id": "group_001",
    "group_name": "Project Discussion Group"
  }'
```

#### 4. Batch Store Group Chat Memory

EverMemOS supports a standardized group chat data format ([GroupChatFormat](data_format/group_chat/group_chat_format.md)). You can use scripts for batch storage:

```bash
# Use script for batch storage
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v3/agentic/memorize

# Validate file format
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --validate-only
```

**GroupChatFormat Example**:

```json
{
  "version": "1.0.0",
  "conversation_meta": {
    "group_id": "group_001",
    "name": "Project Discussion Group",
    "user_details": {
      "user_101": {
        "full_name": "Alice",
        "role": "Product Manager"
      }
    }
  },
  "conversation_list": [
    {
      "message_id": "msg_001",
      "create_time": "2025-02-01T10:00:00+08:00",
      "sender": "user_101",
      "content": "Good morning everyone"
    }
  ]
}
```

For complete format specifications, please refer to [Group Chat Format Specification](data_format/group_chat/group_chat_format.md).

### More Details

For detailed installation, configuration, and usage instructions, please refer to:
- 📚 [Quick Start Guide](docs/dev_docs/getting_started.md) - Complete installation and configuration steps
- 📖 [API Usage Guide](docs/dev_docs/api_usage_guide.md) - API endpoints and data format details
- 🔧 [Development Guide](docs/dev_docs/development_guide.md) - Architecture design and development best practices
- 🚀 [Bootstrap Usage](docs/dev_docs/bootstrap_usage.md) - Script runner usage instructions
- 📝 [Group Chat Format Specification](data_format/group_chat/group_chat_format.md) - Standardized data format

## 📚 Documentation

### Developer Docs
- [Quick Start Guide](docs/dev_docs/getting_started.md) - Installation, configuration, and startup
- [Development Guide](docs/dev_docs/development_guide.md) - Architecture design and best practices
- [Bootstrap Usage](docs/dev_docs/bootstrap_usage.md) - Script runner
- [Dependency Management](docs/dev_docs/project_deps_manage.md) - Package management and version control

### API Documentation
- [Agentic V3 API](docs/api_docs/agentic_v3_api.md) - Agentic layer API
- [Agentic V2 API](docs/api_docs/agentic_v2_api.md) - Agentic layer API (legacy)

### Core Framework
- [Dependency Injection Framework](src/core/di/README.md) - DI container usage guide

### Demos & Evaluation
- [📖 Demo Guide](demo/README.md) - Interactive examples and memory extraction demos
- [📊 Data Guide](data/README.md) - Sample conversation data and format specifications

## 🏗️ Architecture Design

EverMemOS adopts a layered architecture design, mainly including:

- **Agentic Layer**: Memory extraction, vectorization, retrieval, and reranking
- **Memory Layer**: MemCell extraction, episodic memory management
- **Retrieval Layer**: Multi-modal retrieval and result ranking
- **Business Layer**: Business logic and data operations
- **Infrastructure Layer**: Database, cache, message queue adapters, etc.
- **Core Framework**: Dependency injection, middleware, queue management, etc.

For more architectural details, please refer to the [Development Guide](docs/dev_docs/development_guide.md).

## 🤝 Contributing

We welcome all forms of contributions! Whether it's reporting bugs, proposing new features, or submitting code improvements.

Before contributing, please read our [Contributing Guide](CONTRIBUTING.md) to learn about:
- Development environment setup
- Code standards and best practices
- Git commit conventions (Gitemoji)
- Pull Request process

## 🌟 Join Us

<!-- 
This section can include:
- Community communication channels (Discord, Slack, WeChat groups, etc.)
- Technical discussion forums
- Regular meeting information
- Contact email
-->

We are building a vibrant open-source community!

### Contact

- **GitHub Issues**: [Submit issues and suggestions](https://github.com/your-org/memsys_opensource/issues)
- **Discussions**: [Join discussions](https://github.com/your-org/memsys_opensource/discussions)
- **Email**: [Contact email to be added]
- **Community**: [Community link to be added]

### Contributors

Thanks to all the developers who have contributed to this project!

<!-- Can use GitHub Contributors auto-generation -->
<!-- <a href="https://github.com/your-org/memsys_opensource/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=your-org/memsys_opensource" />
</a> -->

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE). This means you are free to use, modify, and distribute this project, with the following key conditions:
- You must include a copy of the Apache 2.0 license
- You must state any significant changes made to the code
- You must retain all copyright, patent, trademark, and attribution notices
- If a NOTICE file is included, you must include it in your distribution

## 🙏 Acknowledgments

<!-- 
This section can include:
- Projects that inspired us
- Open-source libraries used
- Supporting organizations or individuals
-->

Thanks to the following projects and communities for their inspiration and support:

- (To be added)

---

<div align="center">

**If this project helps you, please give us a ⭐️**

Made with ❤️ by the EverMemOS Team

</div>
