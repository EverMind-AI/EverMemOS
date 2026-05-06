<div align="center" id="readme-top">

![banner-gif](https://github.com/user-attachments/assets/0bf97efd-580f-4a53-a2a2-58d6daea7290)

<p align="center">
  <!-- <a href="https://arxiv.org/abs/2601.02163"><img src="https://img.shields.io/badge/arXiv-EverMemOS-F5C842?labelColor=gray&style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv: EverMemOS"></a> -->
  <!-- <a href="https://arxiv.org/abs/2604.08256"><img src="https://img.shields.io/badge/arXiv-HyperMem-F5C842?labelColor=gray&style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv: HyperMem"></a> -->
  <!-- <a href="https://arxiv.org/abs/2602.01313"><img src="https://img.shields.io/badge/arXiv-EverMemBench-F5C842?labelColor=gray&style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv: EverMemBench"></a> -->
  <!-- <a href="https://github.com/EverMind-AI/MSA"><img src="https://img.shields.io/badge/arXiv-Memory%20Sparse%20Attention-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: Memory Sparse Attention"></a> -->
  <a href="https://x.com/evermind"><img src="https://img.shields.io/badge/EverMind-000000?labelColor=gray&style=for-the-badge&logo=x&logoColor=white" alt="X"></a>
  <a href="https://huggingface.co/EverMind-AI"><img src="https://img.shields.io/badge/🤗_HuggingFace-EverMind-F5C842?labelColor=gray&style=for-the-badge" alt="HuggingFace"></a>
  <a href="https://discord.gg/gYep5nQRZJ"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FgYep5nQRZJ%3Fwith_counts%3Dtrue&query=%24.approximate_presence_count&suffix=%20online&label=Discord&color=404EED&labelColor=gray&style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/EverMind-AI/EverOS/discussions/67"><img src="https://img.shields.io/badge/WeCom-EverMind_社区-07C160?labelColor=gray&style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

[Website](https://evermind.ai) · [Documentation](https://docs.evermind.ai) · [Blog](https://evermind.ai/blogs)

</div>


<br>

<details open>
  <summary><kbd>Table of Contents</kbd></summary>

<br>

- [Project Overview](#project-overview)
- [Use Cases](#use-cases)
- [Quick Start](#quick-start)
- [Architecture Methods](#architecture-methods)
- [Benchmarks](#benchmarks)
- [Evaluation](#evaluation)
- [Citations](#citations)
- [Stay Tuned](#stay-tuned)
- [Contributing](#contributing)

<br>

</details>



## Project Overview

**EverOS** is a unified home for building, evaluating, and applying long-term memory in self-evolving agents. The repository is organized around three essential parts:

| Part | What it gives you | Start here |
| :--- | :--- | :--- |
| **Architecture methods** | Memory systems and algorithms you can run, extend, or compare. | [methods/](methods/) |
| **Benchmarks** | Open evaluation suites for memory quality and agent self-evolution. | [benchmarks/](benchmarks/) |
| **Use cases** | Apps, demos, and integrations showing how memory changes real agent workflows. | [use-cases/](use-cases/) |

At the center of EverOS is **EverCore**, a long-term memory operating system for agents. If you are new to the project, scan the use cases first to see what memory enables, then follow the [Quick Start](#quick-start) to run EverCore locally. The architecture and benchmark sections below give you the deeper reference material when you are ready to compare systems or reproduce results.

```
EverOS/
├── benchmarks/                
│   ├── EverMemBench/    
│   └── EvoAgentBench/         
├── methods/                   
│   ├── EverCore/              
│   └── HyperMem/              
└── use-cases/                 
    ├── claude-code-plugin/    
    ├── game-of-throne-demo/   
    ├── openher/            
    ├── ...
    └── ...
```

<br>

## Use Cases

Use cases show what persistent memory makes possible in real products and workflows. Some examples are packaged in this repository; others point to external demos or integrations you can study and adapt.

<table>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/9dcb3dd4-4402-45fa-ae13-e6782f42c7ea)

#### Earth Online Memory Game

Earth Online is a memory-aware productivity game that turns everyday planning into a living quest log.

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/57d8cda7-35a5-4561-b794-5520dffc917b)

#### Multi-Agent Orchestration Platform

Golutra presents a multi-agent workforce for engineering teams, extending the IDE model from a single assistant to coordinated agents.

</td>
</tr>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/e6eaf308-a874-483f-8874-6934bf95a78f)

#### Mobi Companion

An iOS app where users create, nurture, and live with a personalized AI companion called Mobi.

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/9aabcaa9-f97a-49d2-9109-0b5bb696ed41)

#### AI Wearable with Memory

A context-native AI wearable that listens to everyday life and converts conversations into memory.

</td>
</tr>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/df9677ec-386f-4c56-a428-08bca25c54dc)

#### OpenClaw Agent Memory

A 24/7 agent workflow with continuous learning memory across sessions.

[Agent Memory](https://github.com/EverMind-AI/everos/tree/agent_memory) · [Plugin](https://github.com/EverMind-AI/everos/tree/agent_memory/everos-openclaw-plugin)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/3a2357a1-c0c3-464a-8979-0d1cdfc9b0d4)

#### Live2D Character with Memory

Add long-term memory to a real-time Live2D character, powered by [TEN Framework](https://github.com/TEN-framework/ten-framework).

[Code](https://github.com/TEN-framework/ten-framework/tree/main/ai_agents/agents/examples/voice-assistant-with-everos)

</td>
</tr>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/c36bdc04-97d3-4fe9-97d9-4b93b475595a)

#### Computer-Use with Memory

Run screenshot-based analysis with computer-use and store the results in memory.

[Live Demo](https://screenshot-analysis-vercel.vercel.app/)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/54a7cf8f-62c4-4fbc-9d50-b214d034e051)

#### Game of Thrones Memories

A demonstration of AI memory infrastructure through an interactive Q&A experience with *A Game of Thrones*.

[Code](use-cases/game-of-throne-demo)

</td>
</tr>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/af37c1f6-7ba5-430c-b99d-2a7e7eac618f)

#### Claude Code Plugin

Persistent memory for Claude Code. Automatically saves and recalls context from past coding sessions.

[Code](use-cases/claude-code-plugin)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/d521d28c-0ccd-44ff-aecc-828245e2f973)

#### Memory Graph Visualization

Explore stored entities and relationships in a graph interface. Frontend demo; backend integration is in progress.

[Live Demo](https://main.d2j21qxnymu6wl.amplifyapp.com/graph.html)

</td>
</tr>
</table>

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Quick Start

Choose the path that matches your goal:

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
```

| Goal | Component | Entry Point |
| :--- | :--- | :--- |
| Build agents with long-term memory | **EverCore** | [methods/EverCore/](methods/EverCore/) |
| Explore the hypergraph memory architecture | **HyperMem** | [methods/HyperMem/](methods/HyperMem/) |
| Evaluate memory system quality | **EverMemBench** | [benchmarks/EverMemBench/](benchmarks/EverMemBench/) |
| Measure agent self-evolution | **EvoAgentBench** | [benchmarks/EvoAgentBench/](benchmarks/EvoAgentBench/) |
| Adapt an example app or integration | **Use cases** | [use-cases/](use-cases/) |

> Each component has its own installation guide, dependency configuration, and usage examples.

### EverCore

The fastest way to run a memory system locally is to start with EverCore:

```bash
cd methods/EverCore

# Start Docker services
docker compose up -d

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Configure API keys
cp env.template .env
# Edit .env and set:
#   - LLM_API_KEY (for memory extraction)
#   - VECTORIZE_API_KEY (for embedding/rerank)

# Start server
uv run python src/run.py

# Verify installation
curl http://localhost:1995/health
# Expected response: {"status": "healthy", ...}
```

Server runs at `http://localhost:1995` · [Full Setup Guide](methods/EverCore/docs/installation/SETUP.md)

### Basic Usage

Store and retrieve memories with simple Python code:

```python
import requests

API_BASE = "http://localhost:1995/api/v1"

# 1. Store a conversation memory
requests.post(f"{API_BASE}/memories", json={
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+00:00",
    "sender": "user_001",
    "content": "I love playing soccer on weekends"
})

# 2. Search for relevant memories
response = requests.get(f"{API_BASE}/memories/search", json={
    "query": "What sports does the user like?",
    "user_id": "user_001",
    "memory_types": ["episodic_memory"],
    "retrieve_method": "hybrid"
})

result = response.json().get("result", {})
for memory_group in result.get("memories", []):
    print(f"Memory: {memory_group}")
```

[More Examples](methods/EverCore/docs/usage/USAGE_EXAMPLES.md) · [API Reference](https://docs.evermind.ai/api-reference/introduction) · [Interactive Demos](methods/EverCore/docs/usage/DEMOS.md)

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Architecture Methods

These are the memory architectures currently included in EverOS. Use them as runnable systems, research references, or starting points for your own agent memory layer.

<table>
<tr>
<td width="50%" valign="top">

<!-- ![banner-gif](https://github.com/user-attachments/assets/1bbead72-7a6b-4b19-88f2-5bfc8433e3aa) -->

### EverCore

A self-organizing memory operating system inspired by biological imprinting. Extracts, structures, and retrieves long-term knowledge from conversations so agents can remember, understand, and continuously evolve.

LoCoMo **93.05%** · LongMemEval **83.00%**

[Paper](https://arxiv.org/abs/2601.02163) · [Docs](methods/EverCore/)

</td>
<td width="50%" valign="top">

<!-- ![banner-gif](https://github.com/user-attachments/assets/b63d8735-ea94-4ed6-9c0c-a11b55b1a2a4) -->

### HyperMem

A hypergraph-based hierarchical memory architecture that captures high-order associations through hyperedges, with topic, event, and fact layers for coarse-to-fine conversation retrieval.

LoCoMo **92.73%**

[Paper](https://arxiv.org/abs/2604.08256) · [Docs](methods/HyperMem/)

</td>
</tr>
</table>

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Benchmarks

These benchmarks provide shared standards for measuring memory quality and agent self-evolution across systems.

<table>
<tr>
<td width="50%" valign="top">

<!-- ![banner-gif](https://github.com/user-attachments/assets/f6f11c3c-7977-4c3b-8c2b-f7cf13e8f93a) -->

### EverMemBench

Three-layer memory quality evaluation: factual recall, applied reasoning, and personalized generalization.

[Paper](https://arxiv.org/abs/2602.01313) · [Dataset](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) · [Docs](benchmarks/EverMemBench/)

</td>
<td width="50%" valign="top">

<!-- ![banner-gif](https://github.com/user-attachments/assets/79fd03fe-cd6d-4b92-88d7-d66886d31799) -->

### EvoAgentBench

Agent self-evolution evaluation through longitudinal growth curves, transfer efficiency, error avoidance, and skill-hit quality.

[Docs](benchmarks/EvoAgentBench/)

</td>
</tr>
</table>

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Evaluation

Use the evaluation runner to reproduce EverCore results or compare another memory system against the same benchmark tasks.

### Benchmark Results

![EverOS Benchmark Results](https://github.com/user-attachments/assets/41b656e7-6f82-41b7-891d-d6079d10dd39)

### Supported Benchmarks

- **[LoCoMo](https://github.com/snap-research/locomo)** — Long-context memory benchmark with single/multi-hop reasoning
- **[LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)** — Multi-session conversation evaluation
- **[PersonaMem](https://huggingface.co/datasets/bowen-upenn/PersonaMem)** — Persona-based memory evaluation

### Run Evaluations

```bash
cd methods/EverCore

# Install evaluation dependencies
uv sync --group evaluation

# Run smoke test (quick verification)
uv run python -m evaluation.cli --dataset locomo --system everos --smoke

# Run full evaluation
uv run python -m evaluation.cli --dataset locomo --system everos

# View results
cat evaluation/results/locomo-everos/report.txt
```

[Full Evaluation Guide](methods/EverCore/evaluation/README.md) · [Complete Results](https://huggingface.co/datasets/EverMind-AI/everos_Eval_Results)

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Citations

If EverOS helps your research, please cite the relevant paper:

```bibtex
@article{hu2026evermemos,
  title   = {EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning},
  author  = {Chuanrui Hu and Xingze Gao and Zuyi Zhou and Dannong Xu and Yi Bai and Xintong Li and Hui Zhang and Tong Li and Chong Zhang and Lidong Bing and Yafeng Deng},
  journal = {arXiv preprint arXiv:2601.02163},
  year    = {2026}
}

@article{yue2026hypermem,
  title   = {HyperMem: Hypergraph Memory for Long-Term Conversations},
  author  = {Juwei Yue and Chuanrui Hu and Jiawei Sheng and Zuyi Zhou and Wenyuan Zhang and Tingwen Liu and Li Guo and Yafeng Deng},
  journal = {arXiv preprint arXiv:2604.08256},
  year    = {2026}
}

@article{hu2026evaluating,
  title   = {Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues},
  author  = {Chuanrui Hu and Tong Li and Xingze Gao and Hongda Chen and Yi Bai and Dannong Xu and Tianwei Lin and Xiaohong Li and Yunyun Han and Jian Pei and Yafeng Deng},
  journal = {arXiv preprint arXiv:2602.01313},
  year    = {2026}
}
```

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Stay Tuned

Star the repo or join the community links above to follow new architecture methods, benchmark releases, and memory-enabled use cases.

![star us gif](https://github.com/user-attachments/assets/0c512570-945a-483a-9f47-8e067bd34484)

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Contributing

Contributions are welcome across the whole repository: architecture methods, benchmark coverage, use-case examples, documentation, and bug fixes. Browse [Issues](https://github.com/EverMind-AI/EverOS/issues) to find a good entry point, then open a PR when you are ready.

<br>

> [!TIP]
>
> **Welcome all kinds of contributions** 🎉
>
> Help make EverOS better. Code, documentation, benchmark reports, use-case write-ups, and integration examples are all valuable. Share your projects on social media to inspire others.
>
> Connect with one of the EverOS maintainers [@elliotchen200](https://x.com/elliotchen200) on 𝕏 or [@cyfyifanchen](https://github.com/cyfyifanchen) on GitHub for project updates, discussions, and collaboration opportunities.

![divider](https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only)
![divider](https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only)

### Code Contributors

[![EverOS Contributors](https://contrib.rocks/image?repo=EverMind-AI/EverOS)](https://github.com/EverMind-AI/EverOS/graphs/contributors)

![divider](https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only)
![divider](https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only)

### Contribution Guidelines

Read the [Contribution Guidelines](.github/CONTRIBUTING.md) for setup, pull request expectations, and use-case submission notes. For responsible disclosure, see the [Security Policy](.github/SECURITY.md).

![divider](https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only)
![divider](https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only)

### License, Conduct, and Acknowledgments

[Apache 2.0](https://github.com/EverMind-AI/EverOS/blob/main/LICENSE) • [Code of Conduct](.github/CODE_OF_CONDUCT.md) • [Acknowledgments](methods/EverCore/docs/ACKNOWLEDGMENTS.md)

<br>

<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>
