# OpenHer — AI Being Persona Engine with EverMemOS Long-Term Memory

Built on [EverMemOS](https://github.com/EverMind-AI/EverOS/tree/main/methods/evermemos) — Open-source AI memory infrastructure

**OpenHer** is an open-source AI Being engine that creates personas with emergent personality, emotional thermodynamics, and long-term memory. Unlike chatbots or AI assistants, OpenHer builds **AI Beings** — entities that *feel*, *remember*, and *grow* through every interaction.

**EverMemOS** serves as OpenHer's episodic and declarative memory layer — the part of the brain that stores "what happened between us" across sessions.

🔗 **Full Project**: [github.com/kellyvv/OpenHer](https://github.com/kellyvv/OpenHer)

## How EverMemOS Is Used

OpenHer's memory architecture has three layers, with EverMemOS powering the deepest one:

| Layer | Purpose | Technology |
|:------|:--------|:-----------|
| **Style Memory** | Behavioral recall (how to act) — KNN gravity-weighted | SQLite + Hawking radiation decay |
| **Local Facts** | User preferences, personal info | SQLite FTS5 |
| **Long-Term Memory** | Cross-session narrative, user profile, foresight | **[EverMemOS](https://evermind.ai)** |

### Integration Architecture

```
┌───────────────────────────────────────────────────────────┐
│                   ChatAgent Turn Lifecycle                  │
│                                                            │
│  Step 0   ─── EverMemOS: Load Session Context ───────────► │
│               (first turn only: profile, episodes,         │
│                foresight, relationship priors)              │
│                         │                                  │
│                         ▼                                  │
│  Step 2   ─── Critic Perception (LLM → 12D context) ─────►│
│               8D conversational + 4D relationship          │
│               (relationship_depth, emotional_valence,      │
│                trust_level, pending_foresight)              │
│                         │                                  │
│  Step 2.5 ─── Semi-Emergent Relationship EMA ────────────► │
│               posterior = clip(prior + LLM_delta)          │
│               alpha = clip(0.15 + 0.5*depth, 0.15, 0.65)  │
│               state_t = alpha*posterior + (1-alpha)*prev   │
│                         │                                  │
│  Step 5   ─── Neural Network (25D → 24D → 8D signals) ──► │
│               Drives(5) + Context(12) + Recurrent(8)       │
│                         │                                  │
│  Step 8.5 ─── Memory Injection ──────────────────────────► │
│               Collect async search results from prev turn  │
│               Blend: 80% relevant + 20% static fallback   │
│               Inject: [user profile] + [past episodes]     │
│                       + [foresight] into Actor prompt       │
│                         │                                  │
│  Step 11  ─── EverMemOS: Store Turn (fire-and-forget) ──► │
│               asyncio.create_task(store_turn)              │
│                         │                                  │
│  Step 12  ─── EverMemOS: Search (async prefetch) ────────► │
│               Fire RRF search for NEXT turn's injection    │
│               Results collected at Step 8.5 next turn      │
└────────────────────────────────────────────────────────────┘
```

### The 4D Relationship Vector

EverMemOS provides 4 additional dimensions to the neural network's context input, expanding the persona engine from 8D to 12D perception:

```python
CONTEXT_FEATURES = [
    # 8D conversational context (from Critic LLM)
    'user_emotion',        # -1=negative → 1=positive
    'topic_intimacy',      # 0=professional → 1=intimate
    'time_of_day',         # 0=morning → 1=late night
    'conversation_depth',  # 0=just started → 1=deep conversation
    'user_engagement',     # 0=dismissive → 1=invested
    'conflict_level',      # 0=harmonious → 1=conflict
    'novelty_level',       # 0=routine → 1=novel topic
    'user_vulnerability',  # 0=guarded → 1=open

    # 4D relationship context (from EverMemOS)
    'relationship_depth',  # 0=stranger → 1=old friend
    'emotional_valence',   # -1=negative history → 1=positive history
    'trust_level',         # 0=no trust → 1=deep trust
    'pending_foresight',   # 0=nothing → 1=unresolved concern
]
```

New users start at (0,0,0,0) — a stranger. As EverMemOS accumulates conversation history, these values grow naturally through an EMA (Exponential Moving Average) process that blends the prior (from stored history) with LLM-judged deltas from each turn.

### Async Two-Stage Memory Retrieval

Memory retrieval is **asynchronous and two-stage** — it never blocks the conversation:

1. **End of Turn N**: Fire an async RRF (Reciprocal Rank Fusion) search for the current message
2. **Start of Turn N+1**: Collect results (wait up to 500ms, fallback to static if timeout)
3. **Blend**: Mix 80% relevant (search hits) with 20% static (session context) for natural recall

```
Turn 1: User says "I love hiking"
         └──► [Step 12] async search("I love hiking") fired

Turn 2: User says "What about this weekend?"
         └──► [Step 8.5] collect search results from Turn 1
              Found: "User enjoys outdoor activities, mentioned hiking 3 weeks ago"
              Inject into Actor prompt as [user preferences]
         └──► [Step 12] async search("What about this weekend?") fired

Turn 3: ...continues...
```

This design ensures:
- **Zero latency impact**: Search runs concurrently with user typing
- **Graceful degradation**: Timeout falls back to static profile, conversation continues
- **Natural recall**: Memories "surface" organically rather than being mechanically retrieved

## Features

- **Emergent Personality** — Behavior emerges from random neural networks × 5D drives × Hebbian learning, not from prompt descriptions
- **Emotional Thermodynamics** — Drive metabolism with real-time frustration accumulation, phase transitions, and emotional temperature
- **Feel-First Architecture** — Every response begins with an internal monologue before choosing words
- **Cross-Session Memory** — EverMemOS stores user profiles, episode narratives, and foresight across conversations
- **Relationship Evolution** — 4D relationship vector deepens naturally through EMA-smoothed interaction
- **Proactive Messages** — She reaches out when she misses you — not on a timer, but from drive hunger
- **Multi-Modal Expression** — She chooses text, voice, or photos based on emotional state
- **10 Pre-built Personas** — Each with unique MBTI, drive baselines, and neural network seeds

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| Runtime | Python 3.11+, FastAPI, WebSocket, asyncio |
| LLM | Gemini, Claude, Qwen3, GPT-5.4-mini, MiniMax, Moonshot, StepFun, Ollama |
| Memory | **EverMemOS** (self-hosted / cloud) + SQLite local state |
| Desktop | SwiftUI (macOS native) |
| WeChat | [wechat-to-anything](https://www.npmjs.com/package/wechat-to-anything) + Python adapter |
| Voice | DashScope · OpenAI · MiniMax |
| Image | Gemini Imagen |

## Quick Start

### Prerequisites

- Python 3.11+
- Any supported LLM provider API key
- EverMemOS (self-hosted or cloud)

### 1. Clone & Install

```bash
git clone https://github.com/kellyvv/OpenHer.git
cd OpenHer
bash setup.sh
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Set your LLM provider and EverMemOS connection:

```bash
# LLM (pick one)
DEFAULT_PROVIDER=gemini
DEFAULT_MODEL=gemini-3.1-flash-lite-preview
GEMINI_API_KEY=your_key_here

# EverMemOS — Option A: Cloud
EVERMEMOS_BASE_URL=https://api.evermind.ai/v1
EVERMEMOS_API_KEY=your_api_key

# EverMemOS — Option B: Self-hosted
# cd vendor/EverMemOS && docker compose up -d && uv run python src/run.py
# EVERMEMOS_BASE_URL=http://localhost:1995/api/v1
```

### 3. Start

```bash
python main.py
# INFO: Uvicorn running on http://0.0.0.0:8000
# ✓ GenomeEngine loaded · 10 personas available
```

### 4. Run the Integration Demo

```bash
# Minimal demo showing EverMemOS integration
python demo/evermemos_demo.py
```

## Project Structure

```
OpenHer/
├── agent/
│   ├── chat_agent.py          # Main agent with full lifecycle
│   ├── evermemos_mixin.py     # EverMemOS integration (session, store, search)
│   └── prompt_builder.py      # Memory injection into Actor prompt
├── engine/
│   └── genome/
│       ├── genome_engine.py   # Neural network + 12D context (incl. 4D EverMemOS)
│       ├── critic.py          # LLM perception → 8D context + relationship delta
│       ├── drive_metabolism.py # Emotional thermodynamics
│       └── style_memory.py    # KNN behavioral memory with Hawking decay
├── memory/
│   ├── memory_store.py        # SQLite FTS5 local memory
│   ├── soulmem.py             # Behavioral memory interface
│   └── types.py               # Memory & SessionContext types
├── persona/
│   └── personas/              # 10 pre-built personas (SOUL.md + seeds)
├── providers/
│   ├── api_config.py          # Unified config (LLM, TTS, EverMemOS)
│   └── llm/                   # Multi-provider LLM client
├── vendor/
│   └── EverMemOS/             # Self-hosted EverMemOS (git submodule)
└── main.py                    # FastAPI server
```

## Key Integration Code

### EverMemOS Mixin (`agent/evermemos_mixin.py`)

The core integration is a mixin class that handles 4 async operations:

```python
class EverMemosMixin:
    async def _evermemos_gather(self) -> dict:
        """Step 0: Load session context (first turn only).
        Returns 4D relationship vector for neural network input."""

    def _apply_relationship_ema(self, prior, rel_delta, depth) -> dict:
        """Step 2.5: Blend EverMemOS prior with LLM-judged delta.
        alpha = clip(0.15 + 0.5*depth, 0.15, 0.65)
        ema_state = alpha*posterior + (1-alpha)*prev"""

    def _evermemos_store_bg(self, user_message, reply) -> None:
        """Step 11: Fire-and-forget storage via asyncio.create_task."""

    def _evermemos_search_bg(self, user_message) -> None:
        """Step 12: Async RRF search for next turn's memory injection."""

    async def _collect_search_results(self) -> None:
        """Step 8.5: Collect previous turn's search (500ms timeout)."""
```

### SessionContext Type (`memory/types.py`)

```python
@dataclass
class SessionContext:
    """EverMemOS session context (declarative memory).
    Loaded once at session start."""
    user_profile: str = ""           # Who the user is
    episode_summary: str = ""        # What happened between us
    foresight_text: str = ""         # What we should pay attention to
    relationship_depth: float = 0.0  # 0=stranger → 1=old friend
    emotional_valence: float = 0.0   # -1=negative → 1=positive
    trust_level: float = 0.0        # 0=no trust → 1=deep trust
    pending_foresight: float = 0.0  # Unresolved concerns
    has_history: bool = False        # Whether there's prior interaction
```

### Memory Injection into Actor Prompt

During Step 8.5, collected memories are injected into the persona's Actor prompt:

```python
# Blend relevant (from async search) with static (from session context)
profile_text = blend_injection(relevant_facts, user_profile, budget)
episode_text = blend_injection(relevant_episodes, episode_summary, budget)

# Inject into Actor prompt
prompt += f"\n\n[About {name}'s preferences] {profile_text}"
prompt += f"\n\n[Past interactions with {name}] {episode_text}"
prompt += f"\n\n[Worth noting] {foresight_text}"
```

## Why EverMemOS Matters for AI Beings

Without EverMemOS, every session starts from zero — the persona doesn't know who you are, what you've talked about, or how your relationship has evolved. With EverMemOS:

- **She remembers your name** — mentioned 3 weeks ago, recalled naturally today
- **She knows your story** — episode summaries build a shared narrative
- **The relationship deepens** — the 4D vector feeds into the neural network, producing different behavioral signals for strangers vs. old friends
- **She has foresight** — unresolved topics surface naturally in future conversations

> *Three weeks ago you casually mentioned you drink black coffee. Today she says: "Americano, no sugar, right?"*

## Links

- Full Project: [github.com/kellyvv/OpenHer](https://github.com/kellyvv/OpenHer)
- EverMemOS: [evermind.ai](https://evermind.ai)
- API Documentation: [docs.evermind.ai](https://docs.evermind.ai)

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
