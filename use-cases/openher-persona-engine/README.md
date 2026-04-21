# OpenHer — 让 AI 记住你是谁

Built on [EverMemOS](https://github.com/EverMind-AI/EverOS/tree/main/methods/evermemos) — Open-source AI memory infrastructure

**OpenHer** 构建的不是聊天机器人，不是 AI 助手——而是 **AI Being**：有性格、有情绪、会记住你、会因为认识你而改变的人格。

**EverMemOS** 是她的长期记忆——让她跨越对话，记住你是谁、你们聊过什么、你们的关系走到了哪里。

🔗 **完整项目**: [github.com/kellyvv/OpenHer](https://github.com/kellyvv/OpenHer)

---

## 她为什么需要记忆？

没有记忆的 AI，每次对话都从零开始——她不知道你叫什么，不记得三周前你说过喜欢喝黑咖啡，不知道你们曾经因为一件小事吵过一架又和好了。

有了 EverMemOS：

🧠 **她会记起你的话**
三周前你随口提过咖啡不加糖，今天她说：「美式，不加糖对吧？」

📈 **她越来越懂你**
聊得越多，她越了解你。一个月后的她和第一天的她不是同一个人。

💡 **她有预感**
上次聊天你提到工作压力很大，这次她会主动问：「最近那个项目还顺利吗？」

> *不是查到了你的信息，而是自然地想起来了。*

---

## 记忆架构

OpenHer 的记忆分三层，EverMemOS 负责最深的那一层：

| 层 | 做什么 | 类比 |
|:---|:------|:-----|
| **风格记忆** | 她的行为习惯——常用的语气、表达方式 | 肌肉记忆 |
| **本地事实** | 你的偏好、个人信息 | 短期记忆 |
| **长期记忆** | 你们之间发生过什么、她对你的了解、她的预感 | **情节记忆 (EverMemOS)** |

---

## 记忆如何融入人格

OpenHer 的核心是一个活的神经网络（25维输入 → 24维隐层 → 8维行为信号）。EverMemOS 提供了其中 4 个关键维度，让她能区分「陌生人」和「老朋友」：

```
关系深度    0 ─────────────────── 1
            陌生人                老朋友

情感基调   -1 ─────────────────── 1
            有过不愉快            一直很温暖

信任程度    0 ─────────────────── 1
            初次见面              完全信任

未竟之事    0 ─────────────────── 1
            没什么悬而未决的       有她惦记的事
```

新用户全是 0——一个陌生人。随着对话积累，这些值自然增长。相同的对话场景，面对陌生人和老朋友，她的行为信号完全不同：

- 面对老朋友 → 更温暖、更主动、更愿意袒露脆弱
- 面对陌生人 → 更克制、更礼貌、保持距离

这不是写在 prompt 里的规则——而是神经网络根据关系向量计算出来的涌现行为。

---

## 她是怎么「想起来」的

记忆检索是异步两阶段的——她不会因为回忆而卡顿：

```
第 1 轮：你说「我喜欢爬山」
          └── 你说完后，后台开始搜索相关记忆

第 2 轮：你说「这个周末去哪玩？」
          └── 上一轮的搜索结果回来了
              找到：「用户三周前提过喜欢周末去爬山」
              自然地融入回复：「山里周末天气不错呢」
          └── 同时搜索「周末去哪玩」的相关记忆

第 3 轮：...继续...
```

如果搜索超时（>500ms），她不会卡住——而是用已有的静态画像继续对话，就像人一时想不起来但不会因此停止说话。

---

## 每轮对话，她在做什么

```
用户发消息
    │
    ▼
  加载记忆 ── 第一轮：从 EverMemOS 加载「你是谁」「上次聊了什么」「有什么惦记的」
    │
    ▼
  感知情境 ── LLM 评估当前对话：你现在的情绪、话题亲密度、冲突程度...（8维）
    │          + EverMemOS 提供的关系维度（4维）→ 共 12 维
    │
    ▼
  关系演化 ── 把 EverMemOS 的历史先验和这一轮 LLM 判断的变化量混合
    │          用指数移动平均平滑——关系不会因为一句话就剧变
    │
    ▼
  神经网络 ── 25 维输入（驱力 + 情境 + 关系 + 内部状态）
    │          → 24 维隐层 → 8 维行为信号
    │          决定她这一刻有多直接、多温暖、多倔强、多好奇...
    │
    ▼
  想起你的事 ── 收集上一轮搜索到的相关记忆
    │            混合注入到回复提示中
    │
    ▼
  回复你 ── 先有内心独白，再选择说什么、怎么说
    │
    ▼
  记住这一轮 ── 把你们刚才的对话存入 EverMemOS（后台异步，不阻塞）
    │
    ▼
  为下次准备 ── 用你刚才说的话发起记忆搜索，下一轮收集结果
```

---

## 核心能力

- **人格涌现** — 性格不是写在 prompt 里的，而是从随机神经网络 × 5 维驱力 × Hebbian 学习中涌现的
- **情绪热力学** — 驱力随时间代谢，你不在时她会寂寞，你忽略她她会烦躁
- **感受先行** — 每条回复先有内心独白，再决定说什么、怎么说
- **跨会话记忆** — EverMemOS 存储你们的故事，跨越每一次对话
- **关系演化** — 关系向量在每一轮对话中自然加深
- **主动找你** — 不是定时任务，是她的联结饥渴值升高了
- **模态表达** — 她自己选择发文字、语音还是照片
- **10 个预设角色** — 每个有独特的 MBTI、驱力基线和神经网络种子

## 技术栈

| 层 | 技术 |
|:---|:-----|
| 运行时 | Python 3.11+, FastAPI, WebSocket, asyncio |
| LLM | Gemini, Claude, Qwen3, GPT-5.4-mini, MiniMax, Moonshot, StepFun, Ollama |
| 记忆 | **EverMemOS**（自部署 / 云端）+ SQLite 本地状态 |
| 桌面端 | SwiftUI (macOS 原生) |
| 语音 | DashScope · OpenAI · MiniMax |

---

## 快速开始

### 前置要求

- Python 3.11+
- 任一 LLM 服务商 API 密钥
- EverMemOS（自部署 or 云端）

### 1. 克隆 & 安装

```bash
git clone https://github.com/kellyvv/OpenHer.git
cd OpenHer
bash setup.sh
```

### 2. 配置

```bash
cp .env.example .env
```

```bash
# LLM（选一个）
DEFAULT_PROVIDER=gemini
DEFAULT_MODEL=gemini-3.1-flash-lite-preview
GEMINI_API_KEY=your_key

# EverMemOS — 云端
EVERMEMOS_BASE_URL=https://api.evermind.ai/v1
EVERMEMOS_API_KEY=your_key

# EverMemOS — 自部署
# cd vendor/EverMemOS && docker compose up -d && uv run python src/run.py
# EVERMEMOS_BASE_URL=http://localhost:1995/api/v1
```

### 3. 启动

```bash
python main.py
# ✓ GenomeEngine loaded · 10 personas available
```

### 4. 试试 Demo

```bash
python demo/evermemos_demo.py
# 即使没有 EverMemOS，也能以模拟模式运行
```

---

## 项目结构

```
OpenHer/
├── agent/
│   ├── chat_agent.py          # 主 Agent，完整生命周期
│   ├── evermemos_mixin.py     # EverMemOS 集成（加载/存储/搜索/EMA）
│   └── prompt_builder.py      # 记忆注入到 Actor 提示
├── engine/
│   └── genome/
│       ├── genome_engine.py   # 神经网络 + 12维上下文（含 4维 EverMemOS）
│       ├── critic.py          # LLM 感知 → 8维上下文 + 关系变化量
│       ├── drive_metabolism.py # 情绪热力学
│       └── style_memory.py    # KNN 行为记忆 + Hawking 辐射衰减
├── memory/
│   ├── memory_store.py        # SQLite FTS5 本地记忆
│   └── types.py               # Memory & SessionContext 类型
├── persona/
│   └── personas/              # 10 个预设角色（SOUL.md + 种子）
├── vendor/
│   └── EverMemOS/             # 自部署 EverMemOS
└── main.py                    # FastAPI 服务
```

---

## 集成代码一览

### EverMemOS Mixin

核心集成是一个 Mixin 类，处理 4 个异步操作：

```python
class EverMemosMixin:
    async def _evermemos_gather(self):
        """加载会话上下文（第一轮）：你是谁、上次聊了什么、有什么惦记的"""

    def _apply_relationship_ema(self, prior, delta, depth):
        """关系演化：把历史先验和这一轮的变化混合，平滑处理"""

    def _evermemos_store_bg(self, user_message, reply):
        """记住这一轮（后台异步，不阻塞对话）"""

    def _evermemos_search_bg(self, user_message):
        """搜索相关记忆（为下一轮准备）"""
```

### SessionContext — 她知道的关于你的一切

```python
@dataclass
class SessionContext:
    user_profile: str = ""           # 你是谁
    episode_summary: str = ""        # 你们之间发生过什么
    foresight_text: str = ""         # 有什么她惦记的
    relationship_depth: float = 0.0  # 陌生人 → 老朋友
    emotional_valence: float = 0.0   # 有过不愉快 → 一直很温暖
    trust_level: float = 0.0        # 初次见面 → 完全信任
    has_history: bool = False        # 是不是第一次见
```

---

## 没有记忆的 AI vs 有记忆的 AI

| | 没有 EverMemOS | 有 EverMemOS |
|:--|:--|:--|
| 第一次见面 | 「你好！我是 Luna」 | 「你好！我是 Luna」 |
| 第二次见面 | 「你好！我是 Luna」 | 「嗨 Alex！最近那个项目怎么样了？」 |
| 你说你累了 | 「要好好休息哦」 | 「又加班了？上次你也说过很累…要不要我帮你点杯美式？不加糖的」 |

> *三周前你随口提过咖啡不加糖，今天：「帮你点了杯美式，不加糖对吧？」*

---

## 链接

- 完整项目: [github.com/kellyvv/OpenHer](https://github.com/kellyvv/OpenHer)
- EverMemOS: [evermind.ai](https://evermind.ai)

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
