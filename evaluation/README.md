# EverMemOS Evaluation Framework

<p>
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

A unified, modular evaluation framework for benchmarking memory systems on standard datasets.

## Overview

### Evaluation Scope

In addition to **EverMemOS**, this framework supports evaluation of several influential memory systems in the industry:
- **Mem0**
- **MemOS**
- **Zep**
- **MemU**

These systems were selected based on recent industry benchmarks and their prominence in global markets. Since many commercial systems have web-based optimizations not available in their open-source versions, we evaluate them through their **online API interfaces** to ensure fair comparison with production-grade capabilities.

### Implementation

Our adapter implementations are based on:
- **Official open-source repositories**: Mem0, MemOS, Zep on GitHub
- **Official documentation**: Mem0, MemOS, MemU, Zep quick start guide and API documentation
- **Consistent methodology**: All systems evaluated using the same pipeline, datasets, and metrics
- **Unified answer generation**: All systems use **GPT-4.1-mini** as the answer LLM to ensure fair comparison across different memory backends

During our evaluation, we identified several issues in existing open-source reference implementations for benchmarking these systems that could negatively impact their performance. We addressed these implementation gaps to ensure each system is evaluated at its best potential:

- **Mem0 timezone handling**: The latest version returns timestamps in PDT format in search results, requiring additional timezone conversion for accurate temporal reasoning.

- **MemU retrieval enhancement**: While some memories are visible in the backend dashboard, the `/memory/retrieve/related-memory-items` API likely relies on simple vector-based retrieval, which may miss relevant context. Following the official documentation examples, we included category summaries as additional context to improve recall.

- **Zep API migration**: Zep's official open-source evaluation implementation was based on the earlier v2 API. Since Zep has officially upgraded to v3 API, we migrated the evaluation code to v3 following the official documentation to benchmark the latest capabilities.

- **Zep timestamp semantics**: Unlike most memory systems that record conversation timestamps, Zep records event occurrence timestamps. For example, a conversation on March 2nd mentioning "Anna ate a burger yesterday" would be timestamped March 1st, with the memory content preserving the original phrasing. Using standard answer prompts leads to significant errors on temporal questions. Zep's team provides optimized prompts in their open-source evaluation code to handle this. This informed one of our evaluation principles: **each memory system uses its own official answer prompts** rather than a unified prompt template, ensuring fair evaluation of each system's intended usage.


### Evaluation Results
**Results on Locomo**

| Locomo    | single hop | multi hop | temporal | open domain | Overall | Average Tokens | Version                                         | Answer LLM |
|-----------|------------|-----------|----------|-------------|---------|----------------|----------------------------------------------|-----------------|
| Full-context | 94.93      | 90.43     | 87.95    | 71.88       | 91.21   | 20281          |                                              | gpt-4.1-mini    |
| Mem0      | 68.97      | 61.70     | 58.26    | 50.00       | 64.20   | 1016           | web API/v1.0.0 (2025.11)                   | gpt-4.1-mini    |
| Zep       | 90.84      | 81.91     | 77.26    | 75.00       | 85.22   | 1411           | web API/v3 (2025.11)                       | gpt-4.1-mini    |
| MemOS     | 85.37      | 79.43     | 75.08    | 64.58       | 80.76   | 2498           | web API/v1 (2025.11)                       | gpt-4.1-mini    |
| MemU      | 74.91      | 72.34     | 43.61    | 54.17       | 66.67   | 3964           | web API/v1 (2025.11)                      | gpt-4.1-mini    |
| EverMemOS | 96.08      | 91.13     | 89.72    | 70.83       | 92.32   | 2298           | open-source EverMemOS v1.0.0 companion | gpt-4.1-mini    |

*Full-context: using the whole conversation as context for answering questions.


**Results on Longmemeval**

| Longmemeval | Single-session-user  | Single-session-assistant  | Single-session-preference  | Multi-session  | Knowledge-update  | Temporal-reasoning  | Overall |
|-------------|----------------------|---------------------------|----------------------------|----------------|-------------------|---------------------|---------|
| EverMemOS   | 100.00               | 78.57                     | 96.67                      | 78.45          | 87.18             | 71.18               | 82.00   |

> **Note on Reproducibility**: To ensure the reproducibility of our evaluation, we provide full evaluation intermediate data for all methods. You can access the data at [EverMind-AI/EverMemOS_Eval_Results](https://huggingface.co/datasets/EverMind-AI/EverMemOS_Eval_Results).


## Key Features

### Unified & Modular Framework
- **One codebase for all**: No need to write separate code for each dataset or system
- **Plug-and-play systems**: Support multiple memory systems (EverMemOS, Mem0, MemOS, MemU, etc.)
- **Multiple benchmarks**: LoCoMo, LongMemEval, PersonaMem out of the box
- **Consistent evaluation**: All systems evaluated with the same pipeline and metrics

### Automatic Compatibility Detection
The framework automatically detects and adapts to:
- **Multi-user vs Single-user conversations**: Handles both conversation types seamlessly
- **Q&A vs Multiple-choice questions**: Adapts evaluation approach based on question format
- **With/without timestamps**: Works with or without temporal information

### Robust Checkpoint System
- **Cross-stage checkpoints**: Resume from any pipeline stage (add -> search -> answer -> evaluate)
- **Fine-grained resume**: Saves progress every conversation (search) and every 400 questions (answer)
- **Checkpoint file**: `checkpoint_default.json` stores completed stages and intermediate results


## Architecture Overview

### Code Structure

```
evaluation/
├── src/
│   ├── core/           # Pipeline orchestration and data models
│   ├── adapters/       # System-specific implementations
│   │   └── evermemos/  # EverMemOS adapter
│   │       ├── config.py              # ExperimentConfig (all defaults)
│   │       ├── scene_retrieval.py     # Agentic two-level retrieval (entry point)
│   │       ├── retrieval_utils.py     # Base algorithms (BM25, Embedding, RRF, MaxSim)
│   │       ├── stage3_memory_retrivel.py  # Reranker + lightweight retrieval
│   │       └── tools/agentic_utils.py     # Sufficiency check + Multi-Query
│   ├── evaluators/     # Answer evaluation (LLM Judge, Exact Match)
│   ├── converters/     # Dataset format converters
│   └── utils/          # Configuration, logging, I/O
├── config/
│   ├── datasets/       # Dataset configurations (locomo.yaml, etc.)
│   ├── systems/        # System configurations (evermemos.yaml, etc.)
│   └── prompts.yaml    # Prompt templates
├── data/               # Benchmark datasets
└── results/            # Evaluation results and logs
```

### Pipeline Flow

The evaluation consists of 4 sequential stages:

1. **Add**: Ingest conversations, extract MemCells, build indexes (BM25 + Embedding), cluster into Scenes
2. **Search**: Retrieve relevant memories for each question
3. **Answer**: Generate answers using retrieved context
4. **Evaluate**: Assess answer quality with LLM Judge or Exact Match

Each stage saves its output and can be resumed independently.

## Getting Started

### Prerequisites

- Python 3.10+
- EverMemOS environment configured (see main project's `env.template`)

### Default Model Stack

The default configuration (optimized for LoCoMo, 93% accuracy) uses:

| Component | Model | Purpose |
|-----------|-------|---------|
| Answer LLM | `openai/gpt-4.1-mini` (via OpenRouter) | Answer generation, sufficiency check, multi-query |
| Embedding | `Qwen3-Embedding-0.6B` (via SiliconFlow) | Vector search for retrieval |
| Reranker | `Qwen3-Reranker-0.6B` (via SiliconFlow) | Result re-ranking |

These are configured in the main EverMemOS `.env` file. The evaluation framework reuses the same environment variables.

### Dataset Configuration

Dataset configurations are stored in `config/datasets/`. Each dataset has specific settings:

| Dataset | Format | Evaluation | Categories |
|---------|--------|------------|------------|
| LoCoMo | Native | LLM Judge (3 runs) | 1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain, 5=adversarial |
| LongMemEval | Auto-convert | LLM Judge (3 runs) | single-session-user, multi-session, temporal-reasoning, etc. |
| PersonaMem | Auto-convert | Exact Match | recall_user_shared_facts, etc. |

**Filtering Categories:**
```yaml
# config/datasets/locomo.yaml
evaluation:
  filter_category: [5]  # Exclude adversarial questions (category 5)
```

### Data Preparation

Place your dataset files in the `evaluation/data/` directory:

```
evaluation/data/
├── locomo/
│   └── locomo10.json                    # Native LoCoMo format
├── longmemeval/
│   ├── longmemeval_s_cleaned.json       # Original (auto-converts)
│   └── longmemeval_s_locomo_style.json  # Generated
├── personamem/
│   ├── questions_32k.csv                # Original
│   ├── shared_contexts_32k.jsonl        # Original
│   └── personamem_32k_locomo_style.json # Generated
└── personamemv2/
    ├── benchmark/text/
    │   ├── benchmark.csv                # Questions file
    │   └── train.csv                    # Training data
    ├── data/chat_history_32k/
    │   └── chat_history_*.json          # Per-user conversations
    └── personamemv2_32k_locomo_style.json  # Generated
```

**Data Sources:**
- **LoCoMo**: https://github.com/snap-research/locomo/tree/main/data
- **LongMemEval**: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- **PersonaMem v1**: https://huggingface.co/datasets/bowen-upenn/PersonaMem
- **PersonaMem v2**: https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2

The framework auto-converts non-LoCoMo formats on first run.

### Installation

Install evaluation-specific dependencies:

```bash
# For evaluating local systems (EverMemOS)
uv sync --group evaluation

# For evaluating online API systems (Mem0, MemOS, MemU, etc.)
uv sync --group evaluation-full
```

### Environment Configuration

The evaluation framework reuses most environment variables from the main EverMemOS `.env` file:
- `LLM_API_KEY`, `LLM_BASE_URL` (for answer generation with GPT-4.1-mini)
- `VECTORIZE_API_KEY` and  `RERANK_API_KEY` (for embeddings/reranker)

**Important**: For OpenRouter API (used by gpt-4.1-mini), make sure `LLM_API_KEY` is set to your OpenRouter API key (format: `sk-or-v1-xxx`). The system will look for API keys in this order:
1. Explicit `api_key` parameter in config
2. `LLM_API_KEY` environment variable

For testing EverMemOS, please first configure the whole .env file.

**Additional variables for online API systems** (add to `.env` if testing these systems):

```bash
# Mem0
MEM0_API_KEY=your_mem0_api_key

# MemOS
MEMOS_KEY=your_memos_api_key

# MemU
MEMU_API_KEY=your_memu_api_key
```

### Quick Test (Smoke Test)

Run a quick test with limited data to verify everything works:

```bash
# Navigate to project root
cd /path/to/EverMemOS

# Default: first conversation, first 10 messages, first 3 questions
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# Custom: first conversation, 20 messages, 5 questions
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --smoke-messages 20 --smoke-questions 5
```


### CLI Command Reference

```bash
uv run python -m evaluation.cli [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset name (required) | - |
| `--system` | System config name (required) | - |
| `--stages` | Stages to run: add, search, answer, evaluate | All |
| `--smoke` | Enable smoke test mode | False |
| `--smoke-messages` | Messages to process in smoke test | 10 |
| `--smoke-questions` | Questions to test in smoke test | 3 |
| `--from-conv` | Starting conversation index (inclusive) | 0 |
| `--to-conv` | Ending conversation index (exclusive) | None (all) |
| `--run-name` | Run name for distinguishing runs | None |
| `--output-dir` | Custom output directory | Auto-generated |

### Full Evaluation

Run the complete benchmark:

```bash
# Evaluate EverMemOS on LoCoMo (default config, 93% accuracy)
uv run python -m evaluation.cli --dataset locomo --system evermemos

# Evaluate other systems
uv run python -m evaluation.cli --dataset locomo --system memos
uv run python -m evaluation.cli --dataset locomo --system memu
# For mem0, it's recommended to run add first, check the memory status on the web console to make sure it's finished and then following stages.
uv run python -m evaluation.cli --dataset locomo --system mem0 --stages add
uv run python -m evaluation.cli --dataset locomo --system mem0 --stages search answer evaluate

# Evaluate on other datasets
uv run python -m evaluation.cli --dataset longmemeval --system evermemos
uv run python -m evaluation.cli --dataset personamem --system evermemos

# Use --run-name to distinguish multiple runs (useful for A/B testing)
# Results will be saved to: results/{dataset}-{system}-{run-name}/
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name baseline
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name experiment1

# Resume from checkpoint if interrupted (automatic)
# Just re-run the same command - it will detect and resume from checkpoint
uv run python -m evaluation.cli --dataset locomo --system evermemos
```

### View Results

Results are saved to `evaluation/results/{dataset}-{system}[-{run-name}]/`:

```bash
# View summary report
cat evaluation/results/locomo-evermemos/report.txt

# View detailed evaluation results
cat evaluation/results/locomo-evermemos/eval_results.json

# View pipeline execution log
cat evaluation/results/locomo-evermemos/pipeline.log
```

**Result files:**
- `report.txt` - Summary metrics (accuracy, total questions)
- `eval_results.json` - Per-question evaluation details
- `answer_results.json` - Generated answers and retrieved context
- `search_results.json` - Retrieved memories for each question
- `pipeline.log` - Detailed execution logs

## Understanding Results

### Metric

- **Accuracy**: Percentage of correct answers (QA judged by LLM, multiple choice questions judged by exact match)

### Detailed Results

Check `eval_results.json` for per-question breakdown:

**LoCoMo example (Q&A format, evaluated by LLM Judge):**

```json
{
  "total_questions": ...,
  "correct": ...,
  "accuracy": ...,
  "detailed_results": {
      "locomo_exp_user_0": [
         {
            "question_id": "locomo_0_qa0",
            "question": "What is my favorite food?",
            "golden_answer": "Pizza",
            "generated_answer": "Your favorite food is pizza.",
            "judgments": [true, true, true],
            "category": "1"
         }
      ]
  }
}
```

**PersonaMem example (Multiple-choice format, evaluated by Exact Match):**

```json
{
  "overall_accuracy": ...,
  "total_questions": ...,
  "correct_count": ...,
  "detailed_results": [
    {
      "question_id": "acd74206-37dc-4756-94a8-b99a395d9a21",
      "question": "I recently attended an event where ...",
      "golden_answer": "(c)",
      "generated_answer": "(c)",
      "is_correct": true,
      "category": "recall_user_shared_facts"
    }
  ]
}
```

## EverMemOS Retrieval Modes

EverMemOS supports two retrieval modes:

### Agentic Mode (Default, 93% on LoCoMo)

The default `agentic` mode uses a two-level scene retrieval architecture with LLM-guided refinement:

```
┌─────────────────────────────────────────────────────────┐
│ Level 1: Scene Selection                                │
│   Embedding(all) + BM25(all) → RRF Fusion               │
│   → MaxSim Aggregation to Scenes → Top-K Scenes         │
├─────────────────────────────────────────────────────────┤
│ Level 2: Rerank within Scenes                           │
│   All MemCells in scenes → Reranker → Top K              │
│   → LLM Sufficiency Check                               │
│     ├─ Sufficient (68%) → Return results                 │
│     └─ Insufficient (32%) → Round 2                      │
├─────────────────────────────────────────────────────────┤
│ Round 2: Multi-Query (only when insufficient)            │
│   LLM generates 3 refined queries                        │
│   → Each query searches ALL MemCells (not scene-limited) │
│   → Multi-RRF Fusion → Merge with Round 1 → Final Rerank│
└─────────────────────────────────────────────────────────┘
```

**Key design**: Round 2 searches **all MemCells** globally (not limited to selected scenes), ensuring no relevant information is missed.

### Lightweight Mode (BM25-only, Fastest)

The `lightweight` mode uses BM25 keyword matching only, with no LLM calls during retrieval:

```yaml
search:
  mode: "lightweight"
```

| Mode | Method | LLM Calls | Speed | Quality |
|------|--------|-----------|-------|---------|
| `agentic` | Scene + Rerank + LLM Sufficiency + Multi-Query | Yes | Slower | Highest (93%) |
| `lightweight` | BM25 only | No | Fastest | Moderate |


## Custom Configuration Guide

### How It Works

System configurations are YAML files in `config/systems/`. The YAML values override the defaults defined in `ExperimentConfig` (`evaluation/src/adapters/evermemos/config.py`). **Any parameter not specified in YAML uses the default value from `ExperimentConfig`.**

The default `evermemos.yaml` is already optimized for LoCoMo (93% accuracy). You only need custom configs when:
- Evaluating on a different dataset (e.g., LongMemEval needs different clustering params)
- Running ablation studies or hyperparameter experiments
- Changing the model stack

### Creating a Custom Config

```bash
# Copy the default config
cp evaluation/config/systems/evermemos.yaml evaluation/config/systems/my_config.yaml

# Edit my_config.yaml, then run:
uv run python -m evaluation.cli --dataset locomo --system my_config
```

### Configuration Reference

A complete config with all available parameters:

```yaml
# config/systems/my_config.yaml
name: "my_config"
version: "1.0"
description: "My custom configuration"
adapter: "evermemos"

# ─── LLM (Answer generation + Agentic retrieval) ─────────────────────
llm:
  provider: "openai"
  model: "openai/gpt-4.1-mini"          # Answer LLM model
  api_key: "${LLM_API_KEY}"             # Uses env var
  base_url: "${LLM_BASE_URL:https://openrouter.ai/api/v1}"
  temperature: 0.3
  max_tokens: 16384

# ─── Add Stage (MemCell extraction + Clustering) ─────────────────────
add:
  enable_clustering: true                # Required for agentic mode
  enable_scene_retrieval: true           # Build scene index
  enable_foresight_extraction: false
  enable_profile_extraction: false       # Enable for PersonaMem

  # Clustering parameters (dataset-dependent, see examples below)
  cluster_similarity_threshold: 0.70     # Semantic similarity (0-1)
  cluster_max_time_gap_days: 7.0         # Max time gap within a scene

# ─── Search Stage (Retrieval) ────────────────────────────────────────
search:
  mode: "agentic"                        # "agentic" or "lightweight"

  # Level 1: Scene Selection
  scene_top_k: 10                        # Number of scenes to select
  level1_emb_candidates: 50              # Embedding candidates per query
  level1_bm25_candidates: 50             # BM25 candidates per query
  level1_rrf_k: 40                       # RRF fusion constant

  # Level 2: Rerank
  use_reranker: true                     # Use reranker for precision
  response_top_k: 10                     # Round 1 results count

  # Round 2: Multi-Query Expansion (when insufficient)
  use_multi_query: true                  # Enable query expansion
  multi_query_num: 3                     # Refined queries to generate
  round2_response_top_k: 10             # Final results after Round 2
  hybrid_emb_candidates: 50             # Per-query embedding candidates
  hybrid_bm25_candidates: 50            # Per-query BM25 candidates
  hybrid_rrf_k: 40                      # Round 2 RRF constant

# ─── Answer Stage ────────────────────────────────────────────────────
answer:
  max_context_length: 8000
  include_timestamps: true

# ─── Reranker Performance ────────────────────────────────────────────
rerank_batch_size: 32
rerank_max_concurrent: 2
rerank_timeout: 60

# ─── Dataset-specific Overrides (optional) ───────────────────────────
dataset_overrides:
  personamemv2:
    add:
      enable_profile_extraction: true
      profile_extraction_mode: "scene"
      profile_scenario: "assistant"
      profile_min_confidence: 0.6
      profile_min_memcells: 1
      profile_life_max_items: 25
    answer:
      use_profile_in_answer: true
      use_profile_classifier: true
```

### Dataset-Specific Examples

Different datasets have different characteristics and require different clustering parameters:

#### LoCoMo (Default)

LoCoMo contains 10 multi-turn conversations with dense interactions. Use tighter clustering:

```yaml
# The default evermemos.yaml already uses these values
add:
  cluster_similarity_threshold: 0.70     # Tighter: conversations are topically focused
  cluster_max_time_gap_days: 7.0         # Shorter: events are temporally close
search:
  mode: "agentic"
```

#### LongMemEval

LongMemEval has conversations spanning longer time periods with diverse topics. Use looser clustering:

```yaml
# config/systems/evermemos_longmemeval.yaml
name: "evermemos_longmemeval"
version: "1.0"
adapter: "evermemos"

llm:
  provider: "openai"
  model: "openai/gpt-4.1-mini"
  api_key: "${LLM_API_KEY}"
  base_url: "${LLM_BASE_URL:https://openrouter.ai/api/v1}"

add:
  enable_clustering: true
  enable_scene_retrieval: true
  cluster_similarity_threshold: 0.50     # Looser: topics are more diverse
  cluster_max_time_gap_days: 30.0        # Longer: conversations span months

search:
  mode: "agentic"
```

Run with:
```bash
uv run python -m evaluation.cli --dataset longmemeval --system evermemos_longmemeval
```

#### PersonaMem

PersonaMem focuses on user profile recall. Enable profile extraction:

```yaml
# config/systems/evermemos_personamem.yaml
name: "evermemos_personamem"
version: "1.0"
adapter: "evermemos"

llm:
  provider: "openai"
  model: "openai/gpt-4.1-mini"
  api_key: "${LLM_API_KEY}"
  base_url: "${LLM_BASE_URL:https://openrouter.ai/api/v1}"

add:
  enable_clustering: true
  enable_scene_retrieval: true
  enable_profile_extraction: true
  profile_extraction_mode: "scene"
  profile_scenario: "assistant"

search:
  mode: "agentic"

answer:
  use_profile_in_answer: true
  use_profile_classifier: true
```

### Key Parameters to Tune

| Parameter | Section | Default | Range | Impact |
|-----------|---------|---------|-------|--------|
| `cluster_similarity_threshold` | add | 0.70 | 0.50-0.85 | Higher = smaller, tighter scenes |
| `cluster_max_time_gap_days` | add | 7.0 | 1-30 | Shorter = more temporally precise scenes |
| `scene_top_k` | search | 10 | 5-20 | More scenes = higher recall, slower |
| `response_top_k` | search | 10 | 5-20 | More results = more context for LLM |
| `use_multi_query` | search | true | true/false | Disable to test Round 1 only |
| `use_reranker` | search | true | true/false | Disable to skip reranking step |


## Scene Clustering

Scenes are clusters of temporally and semantically related MemCells. They form the "hyperedges" in EverMemOS's hypergraph memory architecture.

```yaml
add:
  enable_clustering: true
  enable_scene_retrieval: true       # Required for agentic mode

  cluster_similarity_threshold: 0.70 # Semantic similarity threshold (0-1)
  cluster_max_time_gap_days: 7.0     # Max time gap within a cluster
```

**Recommended values by dataset:**

| Dataset | `cluster_similarity_threshold` | `cluster_max_time_gap_days` | Rationale |
|---------|-------------------------------|----------------------------|-----------|
| LoCoMo | 0.70 | 7 | Dense, topically focused conversations |
| LongMemEval | 0.50 | 30 | Diverse topics spanning months |
| PersonaMem | 0.70 | 7 | Default (profile-focused evaluation) |

## Advanced Usage

### Run Specific Stages

Skip completed stages to iterate faster:

```bash
# Only run search stage (if add is already done)
uv run python -m evaluation.cli --dataset locomo --system evermemos --stages search

# Run search, answer, and evaluate (skip add)
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --stages search answer evaluate
```
If you have already done search, and you want to do it again, please remove the "search" (and following stages from the completed_stages in the checkpoint_default.json file):
```
  "completed_stages": [
    "answer",
    "search",
    "evaluate",
    "add"
  ]
```

## LLM Judge Evaluation

### Evaluation Mechanism

The LLM Judge evaluates answer correctness using a 3-run voting mechanism:

```json
{
  "llm_judgments": {
    "judgment_1": true,
    "judgment_2": true,
    "judgment_3": false
  }
}
```

An answer is considered **correct** if **majority (>=2/3)** of judgments are `true`.

### Judge Prompt

The judge prompt (configured in `config/prompts.yaml`) is designed to be lenient:
- Format variations are acceptable (e.g., "May 7th" vs "7 May")
- Longer answers containing the key information are correct
- Relative time references matching the gold answer are correct

### Category Breakdown

LoCoMo questions are categorized:

| Category | Type | Description |
|----------|------|-------------|
| 1 | Single-hop | Direct fact retrieval |
| 2 | Multi-hop | Requires connecting multiple facts |
| 3 | Temporal | Time-based reasoning |
| 4 | Open-domain | General knowledge integration |

## Profile Memory Evaluation (PersonaMem)

For PersonaMem dataset, EverMemOS supports Profile memory extraction and retrieval:

### Profile Configuration

```yaml
# config/systems/evermemos.yaml - personamemv2 overrides
dataset_overrides:
  personamemv2:
    add:
      enable_profile_extraction: true
      enable_clustering: true
      profile_extraction_mode: "scene"
      profile_scenario: "assistant"
      profile_min_confidence: 0.6
      profile_min_memcells: 1
      profile_life_max_items: 25
    answer:
      use_profile_in_answer: true
      use_profile_classifier: true
```

### Profile vs Episode Memory Ablation

Compare the contribution of different memory types:

```bash
# Full system (Profile + Episode)
uv run python -m evaluation.cli --dataset personamemv2 --system evermemos

# Only Profile memory
uv run python -m evaluation.cli --dataset personamemv2 --system evermemos_only_profile

# Only Episode memory
uv run python -m evaluation.cli --dataset personamemv2 --system evermemos_only_episode
```

### PersonaMem Evaluation Type

PersonaMem uses **exact match** evaluation for multiple-choice questions:
- Choice extraction: Extracts (a), (b), (c), (d) from generated answers
- Case insensitive matching
- Whitespace normalization

## Retry & Error Handling

### LLM Error Handling

The framework includes robust error handling for LLM API calls:

1. **Answer Stage**: 3 retries with 120s timeout per attempt
2. **Agentic Retrieval**: Automatic retry with temperature increase for JSON parsing failures
3. **Sufficiency Check Fallback**: Assumes "sufficient" on repeated failures (conservative)
4. **Multi-Query Fallback**: Uses original query on expansion failures

### API Rate Limiting

Concurrent requests are controlled per stage:
- **Answer Stage**: Max 50 concurrent
- **Evaluate Stage**: Max 50 concurrent
- **Rerank**: Configurable batch size and concurrency

```yaml
rerank_batch_size: 32
rerank_max_concurrent: 2
rerank_timeout: 60
```

## Output Files Reference

| File | Stage | Description |
|------|-------|-------------|
| `memcells/` | Add | Extracted MemCells per conversation |
| `scenes/` | Add | Scene index (clustered MemCells) |
| `bm25_index/` | Add | BM25 inverted index |
| `vectors/` | Add | Embedding vectors |
| `search_results.json` | Search | Retrieved memories per question |
| `answer_results.json` | Answer | Generated answers with context |
| `eval_results.json` | Evaluate | Per-question judgments |
| `report.txt` | Evaluate | Summary metrics |
| `checkpoint_default.json` | All | Resume checkpoint |
| `pipeline.log` | All | Execution logs |

## License

Same as the parent project.
