# EverMemOS 评估框架
<p>
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

统一的模块化评估框架，用于在标准数据集上对记忆系统进行基准测试。

## 📖 概览

### 评估范围

除了 **EverMemOS** 之外，本框架还支持评估业界几个有影响力的记忆系统：
- **Mem0** 
- **MemOS** 
- **MemU** 
- **Zep** 

我们基于近期业界基准测试和在全球市场的突出地位选择了这些系统。由于许多商业系统在Web版本中包含了开源版本中不可用的优化，我们选择通过它们的**在线 API 接口**进行评估，以确保可以评测到各个系统的最佳水平。

### 实现

我们的适配器实现基于：
- **官方开源仓库**：GitHub 上的 Mem0、MemOS、Zep
- **官方文档**：Mem0、MemOS、MemU、Zep 快速入门指南和 API 文档
- **一致的方法论**：所有系统使用相同的流程、数据集和指标进行评估
- **统一的答案生成模型**：所有系统使用 **GPT-4.1-mini** 作为答案 LLM

在评估过程中，我们发现现有用于评测这些系统的开源参考实现中存在一些可能影响其性能表现的问题。我们对这些实现进行了修复，以确保每个系统都能展现其最佳潜力：

- **Mem0 时区处理**：最新版本在搜索结果中返回的时间戳为 PDT 格式，需要进行额外的时区转换以确保时间推理的准确性。

- **MemU 检索增强**：虽然某些记忆在后台控制面板中可见，但 `/memory/retrieve/related-memory-items` API 可能依赖简单的基于向量的检索，导致遗漏相关上下文。根据官方文档示例，我们将类别摘要作为额外上下文传入以提高召回率。

- **Zep API 迁移**：Zep 的官方开源评估实现基于较早的 v2 API。由于 Zep 已正式升级到 v3 API，我们根据官方文档将评估代码迁移到 v3，以评测最新功能。

- **Zep 时间戳语义**：与大多数记录对话时间戳的记忆系统不同，Zep 记录的是事件发生时间戳。例如，3月2日的对话中提到"安娜昨天吃了汉堡"，时间戳会记为3月1日，而记忆内容保留原始措辞。使用标准答案提示词会导致时间类问题出现大量错误。Zep 团队在其开源评估代码中提供了优化的提示词来处理此问题。这也确立了我们的评估原则之一：**每个记忆系统使用其官方的答案提示词**，而非统一的提示词模板，以确保公平评估每个系统的预期使用方式。

### 评估结果

**Locomo 结果**

| Locomo    | single hop | multi hop | temporal | open domain | Overall | Average Tokens | Version                                         | Answer LLM |
|-----------|------------|-----------|----------|-------------|---------|----------------|----------------------------------------------|-----------------|
| Full-context | 94.93      | 90.43     | 87.95    | 71.88       | 91.21   | 20281          |                                              | gpt-4.1-mini    |
| Mem0      | 68.97      | 61.70     | 58.26    | 50.00       | 64.20   | 1016           | web API/v1.0.0 (2025.11)                   | gpt-4.1-mini    |
| Zep       | 90.84      | 81.91     | 77.26    | 75.00       | 85.22   | 1411           | web API/v3 (2025.11)                       | gpt-4.1-mini    |
| MemOS     | 85.37      | 79.43     | 75.08    | 64.58       | 80.76   | 2498           | web API/v1 (2025.11)                       | gpt-4.1-mini    |
| MemU      | 74.91      | 72.34     | 43.61    | 54.17       | 66.67   | 3964           | web API/v1 (2025.11)                      | gpt-4.1-mini    |
| EverMemOS | 96.08      | 91.13     | 89.72    | 70.83       | 92.32   | 2298           | open-source EverMemOS v1.0.0 companion | gpt-4.1-mini    |

*Full-context: 把完整对话作为上下文，用于回答问题。


**Longmemeval 结果**

| Longmemeval | Single-session-user  | Single-session-assistant  | Single-session-preference  | Multi-session  | Knowledge-update  | Temporal-reasoning  | Overall |
|-------------|----------------------|---------------------------|----------------------------|----------------|-------------------|---------------------|---------|
| EverMemOS   | 100.00               | 78.57                     | 96.67                      | 78.45          | 87.18             | 71.18               | 82.00   |


> **关于复现性的说明**：为了保证测评的可复现性，我们提供了全量的方法的评估中间数据。您可以访问 [EverMind-AI/EverMemOS_Eval_Results](https://huggingface.co/datasets/EverMind-AI/EverMemOS_Eval_Results) 获取数据。



## 🌟 核心特性

### 统一且模块化的框架
- **一个代码库适用于所有场景**：无需为每个数据集或系统编写单独的代码
- **即插即用的系统支持**：支持多种记忆系统（EverMemOS、Mem0、MemOS、MemU 等）
- **多种基准测试**：开箱即用支持 LoCoMo、LongMemEval、PersonaMem
- **一致的评估**：所有系统使用相同的流程和指标进行评估

### 自动兼容性检测
框架会自动检测并适配：
- **多用户 vs 单用户对话**：无缝处理两种对话类型
- **问答 vs 多项选择题**：根据问题格式自适应评估方式
- **有/无时间戳**：支持有或无时间信息的数据

### 强大的检查点系统
- **跨阶段检查点**：可从任何流程阶段恢复（添加 → 搜索 → 回答 → 评估）
- **细粒度恢复**：每个对话（搜索）和每 400 个问题（回答）保存进度
- **检查点文件**：`checkpoint_default.json` 存储已完成阶段和中间结果


## 🏗️ 架构概览

### 代码结构

```
evaluation/
├── src/
│   ├── core/           # 流程编排和数据模型
│   ├── adapters/       # 系统特定实现
│   ├── evaluators/     # 答案评估（LLM 评判、精确匹配）
│   ├── converters/     # 数据集格式转换器
│   └── utils/          # 配置、日志、I/O
├── config/
│   ├── datasets/       # 数据集配置（locomo.yaml 等）
│   ├── systems/        # 系统配置（evermemos.yaml 等）
│   └── prompts.yaml    # 提示词模板
├── data/               # 基准数据集
└── results/            # 评估结果和日志
```

### 流程流转

评估包含 4 个连续阶段：

1. **添加（Add）**：摄取对话并构建索引
2. **搜索（Search）**：为每个问题检索相关记忆
3. **回答（Answer）**：使用检索到的上下文生成答案
4. **评估（Evaluate）**：使用 LLM 评判或精确匹配评估答案质量

每个阶段都会保存其输出，并可独立恢复。

## 🚀 快速开始

### 前置要求

- Python 3.10+
- EverMemOS 环境已配置（参见主项目的 `env.template`）

### 数据集配置

数据集配置存储在 `config/datasets/` 中。每个数据集有特定设置：

| 数据集 | 格式 | 评估方式 | 类别 |
|--------|------|----------|------|
| LoCoMo | 原生 | LLM Judge (3 轮) | 1=单跳, 2=多跳, 3=时间推理, 4=开放域, 5=对抗性 |
| LongMemEval | 自动转换 | LLM Judge (3 轮) | single-session-user, multi-session, temporal-reasoning 等 |
| PersonaMem | 自动转换 | 精确匹配 | recall_user_shared_facts 等 |

**过滤类别：**
```yaml
# config/datasets/locomo.yaml
evaluation:
  filter_category: [5]  # 排除对抗性问题（类别 5）
```

### 数据准备

将数据集文件放置在 `evaluation/data/` 目录中：

```
evaluation/data/
├── locomo/
│   └── locomo10.json                    # 原生 LoCoMo 格式
├── longmemeval/
│   ├── longmemeval_s_cleaned.json       # 原始文件（自动转换）
│   └── longmemeval_s_locomo_style.json  # 生成文件
├── personamem/
│   ├── questions_32k.csv                # 原始文件
│   ├── shared_contexts_32k.jsonl        # 原始文件
│   └── personamem_32k_locomo_style.json # 生成文件
└── personamemv2/
    ├── benchmark/text/
    │   ├── benchmark.csv                # 问题文件
    │   └── train.csv                    # 训练数据
    ├── data/chat_history_32k/
    │   └── chat_history_*.json          # 每用户对话文件
    └── personamemv2_32k_locomo_style.json  # 生成文件
```

**数据来源：**
- **LoCoMo**: https://github.com/snap-research/locomo/tree/main/data
- **LongMemEval**: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- **PersonaMem v1**: https://huggingface.co/datasets/bowen-upenn/PersonaMem
- **PersonaMem v2**: https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2

框架会在首次运行时自动转换非 LoCoMo 格式。

### 安装

安装评估专用依赖：

```bash
# 用于评估本地系统（EverMemOS）
uv sync --group evaluation

# 用于评估在线 API 系统（Mem0、MemOS、MemU 等）
uv sync --group evaluation-full
```

### 环境配置

评估框架重用主 EverMemOS `.env` 文件中的大部分环境变量：
- `LLM_API_KEY`、`LLM_BASE_URL`（用于使用 GPT-4.1-mini 生成答案）
- `VECTORIZE_API_KEY` 和 `RERANK_API_KEY`（用于嵌入向量/重排序）

**⚠️ 重要**：对于 OpenRouter API（由 gpt-4.1-mini 使用），请确保 `LLM_API_KEY` 设置为您的 OpenRouter API 密钥（格式：`sk-or-v1-xxx`）。系统将按以下顺序查找 API 密钥：
1. 配置中的显式 `api_key` 参数
2. `LLM_API_KEY` 环境变量

要测试 EverMemOS，请先配置完整的 .env 文件。

**在线 API 系统的额外变量**（如测试这些系统，请添加到 `.env`）：

```bash
# Mem0
MEM0_API_KEY=your_mem0_api_key

# MemOS
MEMOS_KEY=your_memos_api_key

# MemU
MEMU_API_KEY=your_memu_api_key
```

### 快速测试（冒烟测试）

使用有限数据运行快速测试以验证一切正常：

```bash
# 导航到项目根目录
cd /path/to/memsys-opensource

# 默认：第一个对话，前 10 条消息，前 3 个问题
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# 自定义：第一个对话，20 条消息，5 个问题
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --smoke-messages 20 --smoke-questions 5
```


### CLI 命令参考

```bash
uv run python -m evaluation.cli [OPTIONS]
```

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集名称（必需） | - |
| `--system` | 系统配置名称（必需） | - |
| `--stages` | 运行阶段：add, search, answer, evaluate | 全部 |
| `--smoke` | 启用冒烟测试模式 | False |
| `--smoke-messages` | 冒烟测试中处理的消息数 | 10 |
| `--smoke-questions` | 冒烟测试中测试的问题数 | 3 |
| `--from-conv` | 起始对话索引（包含） | 0 |
| `--to-conv` | 结束对话索引（不包含） | None（全部） |
| `--run-name` | 用于区分多次运行的运行名称 | None |
| `--output-dir` | 自定义输出目录 | 自动生成 |

### 完整评估

运行完整基准测试：

```bash
# 在 LoCoMo 上评估 EverMemOS
uv run python -m evaluation.cli --dataset locomo --system evermemos

# 评估其他系统
uv run python -m evaluation.cli --dataset locomo --system memos
uv run python -m evaluation.cli --dataset locomo --system memu
# 对于 Mem0，建议先运行 add，在 Web 控制台检查记忆状态以确保完成，然后运行后续阶段。
uv run python -m evaluation.cli --dataset locomo --system mem0 --stages add
uv run python -m evaluation.cli --dataset locomo --system mem0 --stages search answer evaluate

# 在其他数据集上评估
uv run python -m evaluation.cli --dataset longmemeval --system evermemos
uv run python -m evaluation.cli --dataset personamem --system evermemos

# 使用 --run-name 区分多次运行（用于 A/B 测试）
# 结果将保存到：results/{dataset}-{system}-{run-name}/
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name baseline
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name experiment1
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name 20241107

# 如果中断则从检查点恢复（自动）
# 只需重新运行相同命令 - 它会检测并从检查点恢复
uv run python -m evaluation.cli --dataset locomo --system evermemos

```

### 查看结果

结果保存到 `evaluation/results/{dataset}-{system}[-{run-name}]/`：

```bash
# 查看摘要报告
cat evaluation/results/locomo-evermemos/report.txt

# 查看详细评估结果
cat evaluation/results/locomo-evermemos/eval_results.json

# 查看流程执行日志
cat evaluation/results/locomo-evermemos/pipeline.log
```

**结果文件：**
- `report.txt` - 摘要指标（准确率、总问题数）
- `eval_results.json` - 每个问题的详细评估
- `answer_results.json` - 生成的答案和检索到的上下文
- `search_results.json` - 每个问题检索到的记忆
- `pipeline.log` - 详细执行日志

## 📊 理解结果

### 指标

- **准确率（Accuracy）**：正确答案的百分比（问答题由 LLM 评判，选择题由规则评判）


### 详细结果

查看 `eval_results.json` 获取每个问题的详细信息：

**LoCoMo 示例（问答格式，由 LLM 评判评估）：**

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

**PersonaMem 示例（多项选择格式，由精确匹配评估）：**

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

## 🔧 高级用法

### 运行特定阶段

跳过已完成的阶段以加快迭代：

```bash
# 仅运行搜索阶段（如果添加已完成）
uv run python -m evaluation.cli --dataset locomo --system evermemos --stages search

# 运行搜索、回答和评估（跳过添加）
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --stages search answer evaluate
```
如果您已经完成了搜索，并希望重新运行，请从 checkpoint_default.json 文件中的 completed_stages 中删除 "search"（以及后续阶段）：
```
  "completed_stages": [
    "answer",
    "search",
    "evaluate",
    "add"
  ]
```


## ⚙️ 自定义配置

复制并修改现有配置来创建自定义系统配置：

```bash
cp evaluation/config/systems/evermemos.yaml evaluation/config/systems/my_config.yaml
uv run python -m evaluation.cli --dataset locomo --system my_config
```

### 配置结构

```yaml
# 基本信息
name: "my_config"
version: "1.0"
adapter: "evermemos"

# 答案生成 LLM
llm:
  provider: "openai"
  model: "openai/gpt-4.1-mini"
  api_key: "${LLM_API_KEY}"           # 使用环境变量
  base_url: "${LLM_BASE_URL:https://openrouter.ai/api/v1}"
  temperature: 0.3
  max_tokens: 16384

# Add 阶段：MemCell 提取与聚类
add:
  enable_clustering: true             # 启用场景聚类
  enable_scene_retrieval: true        # 构建场景索引
  enable_profile_extraction: false    # PersonaMem 需启用
  cluster_similarity_threshold: 0.70  # 聚类阈值 (0-1)
  cluster_max_time_gap_days: 7.0      # 集群内最大时间间隔

# Search 阶段：检索配置
search:
  mode: "agentic"                     # agentic | lightweight
  
  # Level 1: 场景选择
  scene_top_k: 10                     # 选择的场景数
  level1_emb_candidates: 50           # Embedding 候选数
  level1_bm25_candidates: 50          # BM25 候选数
  level1_rrf_k: 40                    # RRF 常数
  
  # Level 2: Agentic 检索
  use_reranker: true
  response_top_k: 10
  use_multi_query: true               # 启用查询扩展
  multi_query_num: 3

# Answer 阶段
answer:
  max_context_length: 8000
  include_timestamps: true

# 数据集特定覆盖
dataset_overrides:
  personamemv2:
    add:
      enable_profile_extraction: true
    answer:
      use_profile_in_answer: true
```

### 关键参数

| 参数 | 所属部分 | 描述 |
|------|----------|------|
| `search.mode` | search | `agentic`（高质量）或 `lightweight`（快速） |
| `scene_top_k` | search | Level 1 选择的场景数量 |
| `cluster_similarity_threshold` | add | 值越高聚类越严格（推荐 0.65-0.80） |
| `cluster_max_time_gap_days` | add | 场景集群内的最大天数间隔 |
| `use_multi_query` | search | 启用 Round 2 查询扩展 |
| `enable_profile_extraction` | add | 提取用户画像（用于 PersonaMem） |

## 🧠 EverMemOS 检索模式

EverMemOS 支持两种检索模式，具有不同的质量-速度权衡：

### Agentic 模式（推荐用于高质量检索）

`agentic` 模式使用两级检索策略，结合 LLM 指导的精炼：

**Level 1：场景选择（RRF + MaxSim）**
1. RRF（倒数排名融合）搜索，结合 Embedding 和 BM25 检索所有 MemCell
2. MaxSim 聚合，将 MemCell 分组到场景中
3. 选择 Top-K 个场景（默认 K=10）

**Level 2：场景内的 Agentic 检索**
1. 对选中场景内的所有 MemCell 进行重排序 → Top 10
2. LLM 充分性检查：判断检索到的上下文是否能回答查询
3. 如果不充分：多查询扩展 + 第二轮搜索

```yaml
# config/systems/locomo_agentic.yaml
search:
  mode: "agentic"
  
  # Level 1：场景选择
  level1_emb_candidates: 50      # Embedding 召回候选数
  level1_bm25_candidates: 50     # BM25 召回候选数
  level1_rrf_k: 40               # RRF 融合常数
  scene_top_k: 10                # 目标场景数
  
  # Level 2：重排序 + Agentic
  use_reranker: true
  response_top_k: 10             # Round 1 返回 Top K
  
  # Round 2：多查询扩展（不充分时触发）
  use_multi_query: true
  multi_query_num: 3             # 扩展查询数量
```

### Lightweight 模式（更快速）

`lightweight` 模式跳过检索阶段的 LLM 调用，以实现更快的评估：

```yaml
search:
  mode: "lightweight"
  lightweight_search_mode: "hybrid"  # 选项：hybrid, bm25_only, emb_only
```

| 模式 | 方法 | 速度 | 质量 |
|------|------|------|------|
| `agentic` | 场景 + 重排序 + LLM | 较慢 | 最高 |
| `lightweight` (hybrid) | RRF (Emb + BM25) | 快速 | 良好 |
| `lightweight` (bm25_only) | 仅 BM25 | 最快 | 中等 |

## 🎯 场景聚类配置

场景是时间和语义相关的 MemCell 集群。在 `add` 部分配置聚类：

```yaml
add:
  enable_clustering: true
  enable_scene_retrieval: true   # agentic 模式必需
  
  cluster_similarity_threshold: 0.70  # 语义相似度阈值（0-1）
  cluster_max_time_gap_days: 7.0      # 集群内最大时间间隔
```

**参数指南：**
- `cluster_similarity_threshold`：值越高 = 聚类越严格（推荐 0.65-0.80）
- `cluster_max_time_gap_days`：时间越短 = 场景越细粒度（推荐 3-14 天）

## ⚖️ LLM Judge 评估

### 评估机制

LLM Judge 使用 3 次运行投票机制评估答案正确性：

```json
{
  "llm_judgments": {
    "judgment_1": true,
    "judgment_2": true,
    "judgment_3": false
  }
}
```

如果**多数（≥2/3）**判断为 `true`，则答案被认为是**正确的**。

### 评判提示词

评判提示词（在 `config/prompts.yaml` 中配置）设计为宽松的：
- 格式变体是可接受的（例如 "May 7th" vs "7 May"）
- 包含关键信息的较长答案是正确的
- 匹配黄金答案的相对时间引用是正确的

### 类别细分

LoCoMo 问题按类别分类：

| 类别 | 类型 | 描述 |
|------|------|------|
| 1 | Single-hop | 直接事实检索 |
| 2 | Multi-hop | 需要连接多个事实 |
| 3 | Temporal | 基于时间的推理 |
| 4 | Open-domain | 通用知识整合 |

## 👤 用户画像记忆评估（PersonaMem）

对于 PersonaMem 数据集，EverMemOS 支持用户画像记忆提取和检索：

### 用户画像配置

```yaml
# config/systems/evermemos.yaml - personamemv2 覆盖配置
dataset_overrides:
  personamemv2:
    add:
      enable_profile_extraction: true      # 启用画像提取
      enable_clustering: true              # 启用场景聚类
      profile_extraction_mode: "scene"     # 从场景上下文提取
      profile_scenario: "assistant"        # 从助手视角构建画像
      profile_min_confidence: 0.6          # 最小置信度阈值
      profile_min_memcells: 1              # 触发提取的最小 MemCell 数
      profile_life_max_items: 25           # 保留的最大画像项数
    answer:
      use_profile_in_answer: true          # 在答案上下文中包含画像
      use_profile_classifier: true         # 使用分类器选择相关画像
```

### 画像 vs 情景记忆消融

比较不同记忆类型的贡献：

```bash
# 完整系统（画像 + 情景）
uv run python -m evaluation.cli --dataset personamemv2 --system evermemos

# 仅画像记忆
uv run python -m evaluation.cli --dataset personamemv2 --system evermemos_only_profile

# 仅情景记忆
uv run python -m evaluation.cli --dataset personamemv2 --system evermemos_only_episode
```

### PersonaMem 评估类型

PersonaMem 使用**精确匹配**评估多项选择题：
- 选项提取：从生成答案中提取 (a), (b), (c), (d)
- 不区分大小写匹配
- 空白符规范化

## 🔄 重试与错误处理

### LLM 错误处理

框架包含 LLM API 调用的健壮错误处理：

1. **Answer 阶段**：每次尝试 120 秒超时，共 3 次重试
2. **Agentic 检索**：JSON 解析失败时自动重试并提高温度
3. **充分性检查回退**：多次失败时假设"充分"（保守策略）
4. **多查询回退**：扩展失败时使用原始查询

### API 速率限制

每个阶段的并发请求受控：
- **Answer 阶段**：最大 50 并发
- **Evaluate 阶段**：最大 50 并发
- **Rerank**：可配置批次大小和并发数

```yaml
rerank_batch_size: 32
rerank_max_concurrent: 2
rerank_timeout: 60
```

## 📁 输出文件参考

| 文件 | 阶段 | 描述 |
|------|------|------|
| `memcells/` | Add | 每个对话提取的 MemCell |
| `bm25_index/` | Add | BM25 倒排索引 |
| `vectors/` | Add | 嵌入向量 |
| `search_results.json` | Search | 每个问题检索到的记忆 |
| `answer_results.json` | Answer | 生成的答案和上下文 |
| `eval_results.json` | Evaluate | 每个问题的评判结果 |
| `report.txt` | Evaluate | 摘要指标 |
| `checkpoint_default.json` | All | 恢复检查点 |
| `pipeline.log` | All | 执行日志 |

## 📄 许可证

与父项目相同。

