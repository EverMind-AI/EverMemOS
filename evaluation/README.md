# EvermemOS Evaluation Framework

A unified, modular evaluation framework for benchmarking memory systems on standard datasets.

## 🌟 Key Features

### Unified & Modular Framework
- **One codebase for all**: No need to write separate code for each dataset or system
- **Plug-and-play systems**: Support multiple memory systems (EvermemOS, mem0, memOS, memU, etc.)
- **Multiple benchmarks**: LoCoMo, LongMemEval, PersonaMem out of the box
- **Consistent evaluation**: All systems evaluated with the same pipeline and metrics

### Automatic Compatibility Detection
The framework automatically detects and adapts to:
- **Multi-user vs Single-user conversations**: Handles both conversation types seamlessly
- **Q&A vs Multiple-choice questions**: Adapts evaluation approach based on question format  
- **With/without timestamps**: Works with or without temporal information

### Robust Checkpoint System
- **Cross-stage checkpoints**: Resume from any pipeline stage (add → search → answer → evaluate)
- **Fine-grained resume**: Saves progress every conversation (search) and every 400 questions (answer)


## 🏗️ Architecture Overview

### Code Structure

```
evaluation/
├── src/
│   ├── core/           # Pipeline orchestration and data models
│   ├── adapters/       # System-specific implementations
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

1. **Add**: Ingest conversations and build indexes
2. **Search**: Retrieve relevant memories for each question
3. **Answer**: Generate answers using retrieved context
4. **Evaluate**: Assess answer quality with LLM Judge or Exact Match

Each stage saves its output and can be resumed independently.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- EvermemOS environment configured (see main project's `env.template`)

### Data Preparation

Place your dataset files in the `evaluation/data/` directory:

**LoCoMo** (native format, no conversion needed):

```
evaluation/data/locomo/
└── locomo10.json
```

**LongMemEval** (auto-converts to LoCoMo format):

```
evaluation/data/longmemeval/
└── longmemeval_s_cleaned.json  # Original file
# → Will auto-generate: longmemeval_s_locomo_style.json
```

**PersonaMem** (auto-converts to LoCoMo format):

```
evaluation/data/personamem/
├── questions_32k.csv           # Original file
└── shared_contexts_32k.jsonl   # Original file
# → Will auto-generate: personamem_32k_locomo_style.json
```

The framework will automatically detect and convert non-LoCoMo formats on first run. You don't need to manually run any conversion scripts.

### Installation

Install evaluation-specific dependencies:

```bash
# For evaluating local systems (EvermemOS)
uv sync --group evaluation

# For evaluating online API systems (mem0, memOS, memU, etc.)
uv sync --group evaluation-full
```

### Environment Configuration

The evaluation framework reuses most environment variables from the main EvermemOS `.env` file:
- `LLM_API_KEY`, `LLM_BASE_URL` (for answer generation)
- `DEEPINFRA_API_KEY` (for embeddings/reranker)

For testing EvermemOS, please first configure the whole .env file.

**Additional variables for online API systems** (add to `.env` if testing these systems):

```bash
# Mem0
MEM0_API_KEY=your_mem0_api_key

# memOS
MEMOS_KEY=your_memos_api_key

# memU
MEMU_API_KEY=your_memu_api_key
```

### Quick Test (Smoke Test)

Run a quick test with limited data to verify everything works:

```bash
# Navigate to project root
cd /path/to/memsys-opensource

# Default: first conversation, first 10 messages, first 3 questions 
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# Custom: first conversation, 20 messages, 5 questions
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --smoke-messages 20 --smoke-questions 5
```


### Full Evaluation

Run the complete benchmark:

```bash
# Evaluate EvermemOS on LoCoMo
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
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name 20241107

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

## 📊 Understanding Results

### Metrics

- **Accuracy**: Percentage of correct answers (as judged by LLM)
- **Total Questions**: Number of questions evaluated
- **Correct**: Number of questions answered correctly

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
            "judgments": [
               true,
               true,
               true
            ],
            "category": "1"
         }
         ...
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
      "question": "I recently attended an event where there was a unique blend of modern beats with Pacific sounds.",
      "golden_answer": "(c)",
      "generated_answer": "(c)",
      "is_correct": true,
      "category": "recall_user_shared_facts"
    }
    ...
  ]
}
```

## 🔧 Advanced Usage

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


### Custom Configuration

Modify system or dataset configurations:

```bash
# Copy and edit configuration
cp evaluation/config/systems/evermemos.yaml evaluation/config/systems/evermemos_custom.yaml
# Edit evermemos_custom.yaml with your changes

# Run with custom config
uv run python -m evaluation.cli --dataset locomo --system evermemos_custom
```


## 🔌 Supported Systems

### Local Systems
- **evermemos** - EvermemOS with MemCell extraction and dual-mode retrieval

### Online API Systems
- **mem0** - Mem0 API
- **memos** - memOS API  
- **memu** - memU HTTP API

## 📚 Supported Datasets

- **locomo** - LoCoMo: Long-term Conversation Memory benchmark
- **longmemeval** - LongMemEval: Extended conversation evaluation
- **personamem** - PersonaMem: Persona consistency evaluation

## 📄 License

Same as the parent project.
