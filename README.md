<div align="center">

<h1>EverMemOS </h1>

<p><strong>Let every interaction be driven by understanding.</strong> · Enterprise-Grade Intelligent Memory System</p>

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-D22128?style=flat-square&logo=apache&logoColor=white" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Supported-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Latest-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img alt="MongoDB" src="https://img.shields.io/badge/MongoDB-7.0+-47A248?style=flat-square&logo=mongodb&logoColor=white" />
  <img alt="Elasticsearch" src="https://img.shields.io/badge/Elasticsearch-8.x-005571?style=flat-square&logo=elasticsearch&logoColor=white" />
  <img alt="Milvus" src="https://img.shields.io/badge/Milvus-2.4+-00A3E0?style=flat-square" />
  <img alt="Redis" src="https://img.shields.io/badge/Redis-7.x-DC382D?style=flat-square&logo=redis&logoColor=white" />
   <a href="https://github.com/EverMind-AI/EverMemOS/releases">
    <img alt="Release" src="https://img.shields.io/badge/release-v1.0.0-4A90E2?style=flat-square" />
  </a>
</p>

<p>
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

</div>

---

> 💬 **More than memory — it's foresight.**

**EverMemOS** is a forward-thinking **intelligent system**.  
While traditional AI memory serves merely as a "look-back" database, EverMemOS enables AI not only to "remember" what happened, but also to "understand" the meaning behind these memories and use them to guide current actions and decisions. In the EverMemOS demo tools, you can see how EverMemOS extracts important information from your history, and then remembers your preferences, habits, and history during conversations, just like a **friend** who truly knows you.
On the **LoCoMo** benchmark, our approach built upon EverMemOS achieved a reasoning accuracy of **92.3%** (evaluated by LLM-Judge), outperforming comparable methods in our evaluation.

---

## 📢 Latest Updates

<table>
<tr>
<td width="100%" style="border: none;">

**[2025-11-02] 🎉 🎉 🎉 EverMemOS v1.0.0 Released!**

- ✨ **Stable Version**: AI Memory System officially open sourced  
- 📚 **Complete Documentation**: Quick start guide and comprehensive API documentation  
- 📈 **Benchmark Testing**: LoCoMo dataset benchmark evaluation pipeline
- 🖥️ **Demo Tools**: Get started quickly with easy-to-use demos

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

**EverMemOS** is an open-source project designed to provide long-term memory capabilities to conversational AI agents. It extracts, structures, and retrieves information from conversations, enabling agents to maintain context, recall past interactions, and progressively build user profiles. This results in more personalized, coherent, and intelligent conversations.

> 📄 **Paper Coming Soon** - Our technical paper is in preparation. Stay tuned!

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

Memory perception layer: quickly recalls relevant memories through multi-round reasoning and intelligent fusion, achieving precise contextual awareness.

#### 🎯 Intelligent Retrieval Tools

- **🧪 Hybrid Retrieval (RRF Fusion)**  
  Parallel execution of semantic and keyword retrieval, seamlessly fused using Reciprocal Rank Fusion algorithm

- **📊 Intelligent Reranking (Reranker)**  
  Batch concurrent processing with exponential backoff retry, maintaining stability under high throughput  
  Reorders candidate memories by deep relevance, prioritizing the most critical information

#### 🤖 Agentic Intelligent Retrieval

- **🎓 LLM-Guided Multi-Round Recall**  
  For insufficient cases, generate 2-3 complementary queries, retrieve and fuse in parallel
  Automatically identifies missing information, proactively filling retrieval blind spots

- **🔀 Multi-Query Parallel Strategy**  
  When a single query cannot fully express intent, generate multiple complementary perspective queries  
  Enhance coverage of complex intents through multi-path RRF fusion

- **⚡ Lightweight Fast Mode**  
  For latency-sensitive scenarios, skip LLM calls and use RRF-fused hybrid retrieval  
  Flexibly balance between speed and quality

#### 🧠 Reasoning Fusion

- **Context Integration**: Concatenate recalled multi-level memories (episodes, profiles, preferences) with current conversation
- **Traceable Reasoning**: Model generates responses based on explicit memory evidence, avoiding hallucination

💡 Through the cognitive loop of **"Structured Memory → Multi-Strategy Recall → Intelligent Retrieval → Contextual Reasoning"**, the AI always "thinks with memory", achieving true contextual awareness.

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
├── demo/                             # Demo code
├── data/                             # Sample conversation data
├── evaluation/                       # Evaluation scripts
│   └── locomo_evaluation/            # LoCoMo benchmark testing
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
- Docker and Docker Compose

### Installation

#### Using Docker for Dependency Services ⭐

Use Docker Compose to start all dependency services (MongoDB, Elasticsearch, Milvus, Redis) with one command:

```bash
# 1. Clone the repository
git clone https://github.com/EverMind-AI/EverMemOS.git
cd memsys_opensource

# 2. Start Docker services
docker-compose up -d

# 3. Verify service status
docker-compose ps

# 4. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 5. Install project dependencies
uv sync

# 6. Configure environment variables
cp env.template .env
# Edit the .env file and fill in the necessary configurations:
#   - LLM_API_KEY: Enter your LLM API Key (for memory extraction)
#   - DEEPINFRA_API_KEY: Enter your DeepInfra API Key (for Embedding and Rerank)
```

**Docker Services**:
| Service | Host Port | Container Port | Purpose |
|---------|-----------|----------------|---------|
| **MongoDB** | 27017 | 27017 | Primary database for storing memory cells and profiles |
| **Elasticsearch** | 19200 | 9200 | Keyword search engine (BM25) |
| **Milvus** | 19530 | 19530 | Vector database for semantic retrieval |
| **Redis** | 6479 | 6379 | Cache service |

> 💡 Use **host ports** when connecting (e.g., `localhost:19200` for Elasticsearch)

> 💡 For detailed Docker configuration and management, see [Docker Deployment Guide](DOCKER_DEPLOYMENT.md)

> 📖 MongoDB detailed installation guide: [MongoDB Installation Guide](docs/usage/MONGODB_GUIDE.md)

---

### How to Use

EverMemOS offers multiple usage methods. Choose the one that best suits your needs:

---

#### 🎯 Run Demo: Memory Extraction and Interactive Chat

The demo showcases the end-to-end functionality of EverMemOS.

---

**🚀 Quick Start: Simple Demo (Recommended)** ⭐

The fastest way to experience EverMemOS! Just 2 steps to see memory storage and retrieval in action:

```bash
# Step 1: Start the API server (in terminal 1)
uv run python src/bootstrap.py src/run.py --port 8001

# Step 2: Run the simple demo (in terminal 2)
uv run python src/bootstrap.py demo/simple_demo.py
```

**What it does:**
- Stores 4 conversation messages about sports hobbies
- Waits 10 seconds for indexing
- Searches for relevant memories with 3 different queries
- Shows complete workflow with friendly explanations

**Perfect for:** First-time users, quick testing, understanding core concepts

See the demo code at [`demo/simple_demo.py`](demo/simple_demo.py)

---

We also provide a full-featured experience:

**Prerequisites: Start the API Server**

```bash
# Terminal 1: Start the API server (required)
uv run python src/bootstrap.py src/run.py --port 8001
```

> 💡 **Tip**: Keep the API server running throughout. All following operations should be performed in another terminal.

---

**Step 1: Extract Memories**

Run the memory extraction script to process sample conversation data and build the memory database:

```bash
# Terminal 2: Run the extraction script
uv run python src/bootstrap.py demo/extract_memory.py
```

This script will:
- Read conversation data from the `data/` directory
- Extract MemCells and save them to the configured database (e.g., MongoDB)
- Generate user profiles and save them to `demo/memcell_outputs/` directory

> **💡 Tip**:
> Configuration is very simple! `extract_memory.py` uses HTTP API and requires only 2 parameters to run.
> For detailed configuration instructions and usage guide, please see the [Demo Documentation](demo/README.md).

**Step 2: Chat with Memory**

After extracting memories, start the interactive chat demo:

```bash
# Terminal 2: Run the chat program (ensure API server is still running)
uv run python src/bootstrap.py demo/chat_with_memory.py
```

This will launch a command-line interface where you can converse with an agent that utilizes the just-extracted memories. For more details on chat features, tips, and suggested questions, please see the [Demo Guide](demo/README.md).

**Interactive Workflow:**
1. **Select Language**: Choose between Chinese (中文) or English interface.
2. **Select Scenario Mode**:
   - **Assistant Mode**: One-on-one conversation with personal memory-based AI assistant.
   - **Group Chat Mode**: Multi-person chat with group memory-based conversation analysis.
3. **Select Conversation Group**: Choose from available groups in your database.
4. **Start Chatting**: Interact with the memory-enhanced AI agent.

---

#### 📊 Run Evaluation: Performance Testing

The evaluation framework provides a unified, modular way to benchmark memory systems on standard datasets (LoCoMo, LongMemEval, PersonaMem).

**Quick Test (Smoke Test)**:

```bash
# Test with limited data to verify everything works
# Default: first conversation, first 10 messages, first 3 questions
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# Custom smoke test: 20 messages, 5 questions
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --smoke-messages 20 --smoke-questions 5

# Test different datasets
uv run python -m evaluation.cli --dataset longmemeval --system evermemos --smoke
uv run python -m evaluation.cli --dataset personamem --system evermemos --smoke

# Test specific stages (e.g., only search and answer)
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --stages search answer

# View smoke test results quickly
cat evaluation/results/locomo-evermemos-smoke/report.txt
```

**Full Evaluation**:

```bash
# Evaluate EvermemOS on LoCoMo benchmark
uv run python -m evaluation.cli --dataset locomo --system evermemos

# Evaluate on other datasets
uv run python -m evaluation.cli --dataset longmemeval --system evermemos
uv run python -m evaluation.cli --dataset personamem --system evermemos

# Use --run-name to distinguish multiple runs (useful for A/B testing)
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name baseline
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name experiment1

# Resume from checkpoint if interrupted (automatic)
# Just re-run the same command - it will detect and resume from checkpoint
uv run python -m evaluation.cli --dataset locomo --system evermemos
```

**View Results**:

```bash
# Results are saved to evaluation/results/{dataset}-{system}[-{run-name}]/
cat evaluation/results/locomo-evermemos/report.txt          # Summary metrics
cat evaluation/results/locomo-evermemos/eval_results.json   # Detailed per-question results
cat evaluation/results/locomo-evermemos/pipeline.log        # Execution logs
```

The evaluation pipeline consists of 4 stages (add → search → answer → evaluate) with automatic checkpointing and resume support.

> **⚙️ Evaluation Configuration**:
> - **Data Preparation**: Place datasets in `evaluation/data/` (see `evaluation/README.md`)
> - **Environment**: Configure `.env` with LLM API keys (see `env.template`)
> - **Installation**: Run `uv sync --group evaluation` to install dependencies
> - **Custom Config**: Copy and modify YAML files in `evaluation/config/systems/` or `evaluation/config/datasets/`
> - **Advanced Usage**: See `evaluation/README.md` for checkpoint management, stage-specific runs, and system comparisons

---

#### 🔌 Call API Endpoints

**Prerequisites: Start the API Server**

Before calling the API, make sure the API server is running:

```bash
# Start the API server
uv run python src/bootstrap.py src/run.py --port 8001
```

> 💡 **Tip**: Keep the API server running throughout. All following API calls should be performed in another terminal.

---

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
    "group_name": "Project Discussion Group",
    "scene": "group_chat"
  }'
```

**API Features**:

- **`/api/v3/agentic/memorize`**: Store single message memory
- **`/api/v3/agentic/retrieve_lightweight`**: Lightweight memory retrieval (Embedding + BM25 + RRF)
- **`/api/v3/agentic/retrieve_agentic`**: Agentic memory retrieval (LLM-guided multi-round intelligent retrieval)

For more API details, please refer to [Agentic V3 API Documentation](docs/api_docs/agentic_v3_api.md).

---

**🔍 Retrieve Memories**

EverMemOS provides two retrieval modes: **Lightweight** (fast) and **Agentic** (intelligent).

**Lightweight Retrieval**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | Yes* | Natural language query (*optional for profile data source) |
| `user_id` | No | User ID |
| `data_source` | Yes | `episode` / `event_log` / `semantic_memory` / `profile` |
| `memory_scope` | Yes | `personal` (user_id only) / `group` (group_id only) / `all` (both) |
| `retrieval_mode` | Yes | `embedding` / `bm25` / `rrf` (recommended) |
| `group_id` | No | Group ID |
| `current_time` | No | Filter valid semantic_memory (format: YYYY-MM-DD) |
| `top_k` | No | Number of results (default: 5) |

**Example 1: Personal Memory**

```bash
curl -X POST http://localhost:8001/api/v3/agentic/retrieve_lightweight \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What sports does the user like?",
    "user_id": "user_001",
    "data_source": "episode",
    "memory_scope": "personal",
    "retrieval_mode": "rrf"
  }'
```

**Example 2: Group Memory**

```bash
curl -X POST http://localhost:8001/api/v3/agentic/retrieve_lightweight \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Discuss project progress",
    "group_id": "project_team_001",
    "data_source": "episode",
    "memory_scope": "group",
    "retrieval_mode": "rrf"
  }'
```

---

**Agentic Retrieval**

LLM-guided multi-round intelligent search with automatic query refinement and result reranking.

```bash
curl -X POST http://localhost:8001/api/v3/agentic/retrieve_agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What foods might the user like?",
    "user_id": "user_001",
    "group_id": "chat_group_001",
    "top_k": 20,
    "llm_config": {
      "model": "gpt-4o-mini",
      "api_key": "your_api_key"
    }
  }'
```

> ⚠️ Agentic retrieval requires LLM API key and takes longer, but provides higher quality results for queries requiring multiple memory sources and complex logic.

> 📖 Full Documentation: [Agentic V3 API](docs/api_docs/agentic_v3_api.md) | Testing Tool: `demo/tools/test_retrieval_comprehensive.py`

---

#### 📦 Batch Store Group Chat Memory

EverMemOS supports a standardized group chat data format ([GroupChatFormat](data_format/group_chat/group_chat_format.md)). You can use scripts for batch storage:

```bash
# Use script for batch storage (Chinese data)
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_zh.json \
  --api-url http://localhost:1995/api/v3/agentic/memorize \
  --scene group_chat 

# Or use English data
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_en.json \
  --api-url http://localhost:1995/api/v3/agentic/memorize \
  --scene group_chat

# Validate file format
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_en.json \
  --scene group_chat \
  --validate-only
```

> ℹ️ `scene` is required and must be either `assistant` or `group_chat`, so the memorization pipeline knows which extraction profile to run.

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
- [📊 Evaluation Guide](evaluation/locomo_evaluation/README.md) - Testing EverMemOS-based methods on the public LoCoMo dataset

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

- **GitHub Issues**: [Submit issues and suggestions](https://github.com/EverMind-AI/EverMemOS/issues)
- **Discussions**: [Join discussions](https://github.com/EverMind-AI/EverMemOS/discussions)
- **Email**: evermind@shanda.com
- **Reddit**: [r/EverMindAI](https://www.reddit.com/r/EverMindAI/)
- **X**: [@EverMindAI](https://x.com/EverMindAI)

### Contributors

Thanks to all the developers who have contributed to this project!

<!-- Can use GitHub Contributors auto-generation -->
<!-- <a href="https://github.com/your-org/memsys_opensource/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=your-org/memsys_opensource" />
</a> -->

## 📖 Citation

If you use EverMemOS in your research, please cite our paper (coming soon):

```
Coming soon
```

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

- [Memos](https://github.com/usememos/memos) - Thank you to the Memos project for providing a comprehensive, standardized open-source note-taking service that has provided valuable inspiration for our memory system design.

---

<div align="center">

**If this project helps you, please give us a ⭐️**

Made with ❤️ by the EverMemOS Team

</div>
