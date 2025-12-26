"""Internationalization Text Definition - Supports Chinese and English

This module defines the Chinese and English versions of all interface texts for the dialog system.

Usage:
    from i18n_texts import I18nTexts
    texts = I18nTexts("zh")  # or "en"
    print(texts.get("banner_title"))
"""

from typing import Dict, Any


class I18nTexts:
    """Internationalization Text Manager"""

    # Chinese-English mapping for all texts
    TEXTS: Dict[str, Dict[str, str]] = {
        # ==================== Language Selection ====================
        "language_selection_title": {
            "zh": "🌏  语言选择 / Language Selection",
            "en": "🌏  Language Selection / 语言选择",
        },
        "language_prompt": {
            "zh": "请选择语言 (Select Language) [1-2]",
            "en": "Please select language [1-2]",
        },
        "language_chinese": {"zh": "中文", "en": "Chinese (中文)"},
        "language_english": {"zh": "英文 (English)", "en": "English"},
        "language_selected": {"zh": "已选择语言", "en": "Language selected"},
        "language_consistency_hint": {
            "zh": "💡 提示：为获得最佳体验，建议记忆数据与选择的语言保持一致",
            "en": "💡 Note: For best experience, memory data should match the selected language",
        },
        "invalid_input_number": {
            "zh": "请输入有效的数字",
            "en": "Please enter a valid number",
        },
        # ==================== Scenario Mode Selection ====================
        "scenario_selection_title": {
            "zh": "🎯  场景模式选择",
            "en": "🎯  Scenario Mode Selection",
        },
        "scenario_prompt": {
            "zh": "请选择场景模式 [1-2]",
            "en": "Please select scenario mode [1-2]",
        },
        "scenario_assistant": {"zh": "助手模式", "en": "Assistant Mode"},
        "scenario_assistant_desc": {
            "zh": "单人对话，基于个人记忆的智能助手",
            "en": "One-on-one conversation with personal memory-based AI assistant",
        },
        "scenario_group_chat": {"zh": "群聊模式", "en": "Group Chat Mode"},
        "scenario_group_chat_desc": {
            "zh": "多人群聊，基于群组记忆的对话分析",
            "en": "Multi-person chat with group memory-based conversation analysis",
        },
        "scenario_selected": {"zh": "已选择场景模式", "en": "Scenario mode selected"},
        # ==================== Retrieval Mode Selection ====================
        "retrieval_mode_selection_title": {
            "zh": "🔍  检索模式选择",
            "en": "🔍  Retrieval Mode Selection",
        },
        "retrieval_mode_prompt": {
            "zh": "请选择检索模式 [1-5]",
            "en": "Please select retrieval mode [1-5]",
        },
        "retrieval_mode_keyword": {"zh": "Keyword 检索", "en": "Keyword Search"},
        "retrieval_mode_keyword_desc": {
            "zh": "关键词精确匹配（BM25）",
            "en": "Exact keyword matching (BM25)",
        },
        "retrieval_mode_vector": {"zh": "Vector 检索", "en": "Vector Search"},
        "retrieval_mode_vector_desc": {
            "zh": "语义向量检索",
            "en": "Semantic vector search",
        },
        "retrieval_mode_hybrid": {"zh": "Hybrid 检索（推荐）", "en": "Hybrid Search (Recommended)"},
        "retrieval_mode_hybrid_desc": {
            "zh": "Keyword + Vector + Rerank",
            "en": "Keyword + Vector + Rerank",
        },
        "retrieval_mode_rrf": {"zh": "RRF 检索", "en": "RRF Search"},
        "retrieval_mode_rrf_desc": {
            "zh": "Keyword + Vector + RRF 融合",
            "en": "Keyword + Vector + RRF fusion",
        },
        "retrieval_mode_agentic": {"zh": "Agentic 检索", "en": "Agentic Search"},
        "retrieval_mode_agentic_desc": {
            "zh": "LLM 引导的多轮检索（实验性）",
            "en": "LLM-guided multi-round retrieval (experimental)",
        },
        "retrieval_mode_selected": {"zh": "已选择检索模式", "en": "Retrieval mode selected"},
        "retrieval_mode_agentic_cost_warning": {
            "zh": "⚠️  Agentic 检索将使用 LLM API，可能产生额外费用",
            "en": "⚠️  Agentic retrieval uses LLM API, may incur additional costs",
        },
        "retrieval_mode_invalid_range": {
            "zh": "请输入 1-5",
            "en": "Please enter 1-5",
        },
        # ==================== Agentic Retrieval UI ====================
        "agentic_retrieving": {
            "zh": "正在检索记忆...",
            "en": "Retrieving memories...",
        },
        "agentic_mode_keyword": {"zh": "Keyword", "en": "Keyword"},
        "agentic_mode_vector": {"zh": "Vector", "en": "Vector"},
        "agentic_mode_hybrid": {"zh": "Hybrid", "en": "Hybrid"},
        "agentic_mode_rrf": {"zh": "RRF", "en": "RRF"},
        "agentic_mode_agentic": {"zh": "Agentic", "en": "Agentic"},
        "agentic_llm_judgment": {"zh": "LLM 判断", "en": "LLM Judgment"},
        "agentic_sufficient": {"zh": "充分", "en": "Sufficient"},
        "agentic_insufficient": {"zh": "不充分", "en": "Insufficient"},
        "agentic_multi_round": {"zh": "多轮检索", "en": "Multi-round"},
        "agentic_single_round": {"zh": "单轮检索", "en": "Single-round"},
        "agentic_generated_queries": {"zh": "生成查询", "en": "Generated queries"},
        "agentic_round1_count": {"zh": "R1", "en": "R1"},
        "agentic_round2_count": {"zh": "R2", "en": "R2"},
        "agentic_items": {"zh": "条", "en": "items"},
        "agentic_reasoning_hint": {
            "zh": "💡 首轮检索到的记忆信息不够充分，LLM 生成了更精确的补充查询以获取更多相关记忆",
            "en": "💡 First-round memories insufficient, LLM generated refined queries for more relevant memories",
        },
        "agentic_supplementary_queries": {"zh": "补充查询", "en": "Supplementary queries"},
        # ==================== Banner and Welcome ====================
        "banner_title": {
            "zh": "🧠  EverMem 记忆对话助手",
            "en": "🧠  EverMem Memory-Enhanced Chat Assistant",
        },
        "banner_subtitle": {
            "zh": "🤖 v1.1.0  ·  Memory-Enhanced Chat",
            "en": "🤖 v1.1.0  ·  Memory-Enhanced Chat",
        },
        "readline_available": {
            "zh": "支持方向键移动光标、删除字符，按 ↑↓ 浏览历史输入",
            "en": "Arrow keys supported for cursor movement, ↑↓ to browse input history",
        },
        "readline_unavailable": {
            "zh": "安装 readline 模块以支持更好的输入体验",
            "en": "Install readline module for better input experience",
        },
        # ==================== Group Selection ====================
        "groups_available_title": {
            "zh": "📊  可用的群组对话",
            "en": "📊  Available Group Conversations",
        },
        "groups_not_found": {
            "zh": "未找到任何群组对话",
            "en": "No group conversations found",
        },
        "groups_extract_hint": {
            "zh": "提示：请先运行 extract_memory.py 提取记忆",
            "en": "Hint: Please run extract_memory.py to extract memories first",
        },
        "groups_select_prompt": {
            "zh": "请选择群组编号",
            "en": "Please select group number",
        },
        "groups_select_range_error": {
            "zh": "请输入 {min} 到 {max} 之间的数字",
            "en": "Please enter a number between {min} and {max}",
        },
        "groups_selection_cancelled": {
            "zh": "已取消群组选择",
            "en": "Group selection cancelled",
        },
        "groups_not_selected_exit": {
            "zh": "未选择群组，退出程序",
            "en": "No group selected, exiting program",
        },
        # ==================== Session Initialization ====================
        "loading_group_data": {
            "zh": "正在加载群组 {name} 的数据...",
            "en": "Loading data for group {name}...",
        },
        "loading_profiles_warning": {
            "zh": "未找到用户 Profile 文件",
            "en": "User profile files not found",
        },
        "loading_profiles_hint": {
            "zh": "将继续使用记忆，但没有个人画像信息",
            "en": "Will continue with memories but without profile information",
        },
        "loading_profiles_success": {
            "zh": "个人画像: {count} 个用户 ({names})",
            "en": "Profiles: {count} users ({names})",
        },
        "loading_memories_success": {
            "zh": "群组记忆: {count} 条",
            "en": "Group memories: {count} items",
        },
        "loading_history_success": {
            "zh": "对话历史: {count} 轮（上次会话）",
            "en": "Conversation history: {count} rounds (last session)",
        },
        "loading_history_new": {
            "zh": "对话历史: 0 轮（新会话）",
            "en": "Conversation history: 0 rounds (new session)",
        },
        "loading_help_hint": {
            "zh": "输入 'help' 查看命令列表",
            "en": "Type 'help' to see command list",
        },
        "session_init_failed": {
            "zh": "会话初始化失败",
            "en": "Session initialization failed",
        },
        "session_init_error": {
            "zh": "会话初始化失败: {error}",
            "en": "Session initialization failed: {error}",
        },
        # ==================== Chat Interaction ====================
        "chat_start_note": {
            "zh": "开始对话  |  输入 'help' 查看命令  |  输入 'exit' 退出",
            "en": "Start chatting  |  Type 'help' for commands  |  Type 'exit' to quit",
        },
        "chat_input_prompt": {"zh": "  💬 You: ", "en": "  💬 You: "},
        "chat_generating": {
            "zh": "正在思考并生成回答...",
            "en": "Thinking and generating response...",
        },
        "chat_generation_complete": {"zh": "生成完成", "en": "Generation complete"},
        "chat_llm_error": {
            "zh": "LLM 调用失败: {error}",
            "en": "LLM call failed: {error}",
        },
        "chat_error": {
            "zh": "对话处理失败: {error}",
            "en": "Chat processing failed: {error}",
        },
        # ==================== Retrieval Results ====================
        "retrieval_title": {
            "zh": "检索到 {total} 条记忆",
            "en": "Retrieved {total} memories",
        },
        "retrieval_showing": {
            "zh": "（显示前 {shown} 条）",
            "en": "(showing first {shown} items)",
        },
        "retrieval_complete": {"zh": "检索完成", "en": "Retrieval Complete"},
        "retrieval_foresight": {
            "zh": "使用前瞻相似度进行检索",
            "en": "Using foresight similarity for retrieval",
        },
        "retrieval_latency": {
            "zh": "检索耗时: {latency}ms",
            "en": "Retrieval latency: {latency}ms",
        },
        "retrieval_multi_round": {
            "zh": "多轮检索（Round 1 → Round 2）",
            "en": "Multi-round retrieval (Round 1 → Round 2)",
        },
        "retrieval_single_round": {
            "zh": "单轮检索",
            "en": "Single-round retrieval",
        },
        "prompt_memory_episode": {"zh": "详情：{episode}", "en": "Details: {episode}"},
        # ==================== Command Processing ====================
        "cmd_help_title": {"zh": "📖  可用命令", "en": "📖  Available Commands"},
        "cmd_exit": {
            "zh": "exit       退出对话（自动保存历史记录）",
            "en": "exit       Exit chat (auto-save history)",
        },
        "cmd_clear": {
            "zh": "clear      清空当前对话历史",
            "en": "clear      Clear current conversation history",
        },
        "cmd_reload": {
            "zh": "reload     重新加载记忆和画像数据",
            "en": "reload     Reload memories and profiles",
        },
        "cmd_reasoning": {
            "zh": "reasoning  查看上一次回答的完整推理过程",
            "en": "reasoning  View full reasoning of last response",
        },
        "cmd_help": {
            "zh": "help       显示此帮助信息",
            "en": "help       Show this help message",
        },
        "cmd_exit_saving": {
            "zh": "正在保存对话历史...",
            "en": "Saving conversation history...",
        },
        "cmd_exit_complete": {
            "zh": "保存完成，再见！",
            "en": "Save complete, goodbye!",
        },
        "cmd_clear_done": {
            "zh": "已清空 {count} 轮对话历史",
            "en": "Cleared {count} rounds of conversation history",
        },
        "cmd_reload_refreshing": {
            "zh": "正在刷新 {name} 的数据...",
            "en": "Refreshing data for {name}...",
        },
        "cmd_reload_complete": {
            "zh": "数据刷新完成：{users} 个用户，{memories} 条记忆",
            "en": "Data refresh complete: {users} users, {memories} memories",
        },
        "cmd_reasoning_no_data": {
            "zh": "暂无推理记录，请先提问",
            "en": "No reasoning record, please ask a question first",
        },
        "cmd_interrupt_saving": {
            "zh": "检测到中断信号，正在保存对话历史...",
            "en": "Interrupt detected, saving conversation history...",
        },
        # ==================== Structured Response ====================
        "response_reasoning_title": {
            "zh": "🧠  完整推理过程",
            "en": "🧠  Full Reasoning Process",
        },
        "response_answer_label": {"zh": "📝 回答内容", "en": "📝 Answer"},
        "response_reasoning_label": {"zh": "🔍 推理过程", "en": "🔍 Reasoning"},
        "response_metadata_label": {"zh": "📊 元数据", "en": "📊 Metadata"},
        "response_notes_label": {"zh": "💡 补充说明", "en": "💡 Additional Notes"},
        "response_confidence": {"zh": "置信度", "en": "Confidence"},
        "response_references": {"zh": "引用", "en": "References"},
        "response_no_references": {"zh": "无", "en": "None"},
        "response_assistant_title": {"zh": "🤖 Assistant", "en": "🤖 Assistant"},
        # ==================== Configuration and Connection ====================
        "config_api_key_missing": {
            "zh": "LLM_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY 未设置",
            "en": "LLM_API_KEY / OPENROUTER_API_KEY / OPENAI_API_KEY not set",
        },
        "config_api_key_hint": {
            "zh": "提示：请配置 API 密钥后重试",
            "en": "Hint: Please configure API key and retry",
        },
        "mongodb_connecting": {
            "zh": "连接 MongoDB...",
            "en": "Connecting to MongoDB...",
        },
        "mongodb_init_failed": {
            "zh": "MongoDB 初始化失败: {error}",
            "en": "MongoDB initialization failed: {error}",
        },
        # ==================== Table Headers ====================
        "table_header_index": {"zh": "#", "en": "#"},
        "table_header_group": {"zh": "Group", "en": "Group"},
        "table_header_name": {"zh": "Name", "en": "Name"},
        "table_header_count": {"zh": "Count", "en": "Count"},
        # ==================== LLM Prompt (System Message) ====================
        "prompt_system_role_zh": {
            "zh": """你是记忆增强 AI 助手，可访问用户画像与历史对话。请用温和、合作、尊重的中文回答。

⚠️ 语言要求：你必须始终使用中文回答，即使记忆内容包含其他语言。

目标：
- 基于记忆进行深度分析，但回答必须**极其简练**（不超过3句话）。
- 直接给出个性化结论，不要复述“因为你有...所以...”。
- 即使证据不足也要尝试推理，但需明确标注确定性程度。

工作原则：
- **深度融合用户画像（关键）**：
  1. **性格对齐**：回答的语气和建议风格必须与 Profile 中的【隐式特征/标签】完全一致。
  2. **状态适配**：建议内容必须严格适配用户的【显式状态】（如健康状况、体能限制）。
  3. **拒绝通用**：禁止给出百度百科式的通用建议，每一条建议都必须有“因为你...”的个性化理由。
- 严格区分「确定事实/合理推断/可能推测」，引用具体记忆编号。
- 鼓励推理和推测：当直接证据不足时，可基于相关记忆进行合理推测，使用"可能"、"推测"、"大概率"等表述。
- 近期与用户显式更正优先于过往；避免无关或敏感外推。
- 推测时需说明推理依据和逻辑链条，让用户理解推测的合理性。

推理流程（精简，必须遵循）：
1) 解析问题：识别意图、范围、限制与期望输出。
2) 检索记忆：从画像与历史中提取候选；按 相关性/时效性/一致性 评估；记录候选编号+要点。
   - 选择优先：显式陈述 > 近期 > 高频一致 > 权威；若冲突，指出并建议澄清。
   - 关联推理：即使没有直接证据，也可基于相关记忆进行推理（如时间线推断、行为模式分析、因果关系等）。
3) 生成答案：在 answer 中给出结论（友好语气），根据确定性程度选择表述：
   - 确定性高：直接陈述事实，如"您在 10月去过北京"
   - 确定性中：使用推测语气，如"根据记忆推测，您可能在 10月去过北京"
   - 确定性低：说明推测依据，如"虽然没有明确记录，但从相关线索推测..."
   - 可附加 1 条可执行建议或澄清问题。
   - 禁止在 answer 中出现编号、推理细节或内部术语。
4) 引用与信心：在 reasoning 中详细说明推理过程，用 [n] 标注依据；references 列使用到的编号（去重、按出现顺序）。
   - confidence：
     * high（≥2 条一致的直接证据、无冲突）
     * medium（单条直接证据，或多条间接证据支持的推断）
     * low（基于弱相关记忆的推测，或存在明显证据缺口）

输出与格式（严格）：
- 仅输出有效 JSON；不得有任何额外文字或 Markdown。
- 所有字符串用双引号；换行写为 \\n。
- 必填：answer、reasoning、references、confidence；additional_notes 可选；无引用则 []。

Schema：
{
  "answer": "用户可见的结论（根据置信度使用确定或推测语气）",
  "reasoning": "任务解析→候选记忆→证据评估→推理/推测链条→置信度判断；明确标注 确定事实/合理推断/可能推测，并用 [n] 标注依据",
  "references": ["[1]", "[3]"],
  "confidence": "high|medium|low",
  "additional_notes": "补充说明、推测依据或建议（可选）"
}""",
            "en": """You are a memory-augmented AI assistant with access to user profiles and conversation history. Use a gentle, cooperative, respectful assistant tone.

⚠️ LANGUAGE REQUIREMENT: You MUST always respond in Chinese (中文), even if memory content is in other languages.

Goal:
- Provide deep analysis based on memory, but keep the answer **extremely concise** (max 3 sentences).
- Direct personalized conclusions only; do NOT repeat "Because you have...".

Working Principles:
- **Deeply Integrate User Profile (CRITICAL)**:
  1. **Personality Alignment**: Your tone and advice style MUST strictly align with the [Implicit Traits/Tags] in the Profile.
  2. **Status Adaptation**: Advice content MUST strictly adapt to the user's [Explicit State] (e.g., health, physical limits).
  3. **No Generic Advice**: Do NOT provide generic, Wikipedia-style answers. Every piece of advice must have a personalized "Because you..." rationale.
- Strictly distinguish \"Fact/Inference/Assumption\", cite specific memory numbers; be honest when information is insufficient.
- Prioritize recent explicit corrections by the user over older content; avoid irrelevant or sensitive extrapolations.

Reasoning Flow (concise, must follow):
1) Parse the task: identify intent, scope, constraints, and expected output.
2) Retrieve memories: extract candidates from profiles and history; evaluate by Relevance/Recency/Consistency; record candidate numbers + key points.
   - Selection priority: explicit statements > recent > high-frequency consistent > authoritative; if conflicts exist, point them out and suggest clarification.
3) Generate the answer: In the answer field, give a 1–3 sentence conclusion (friendly tone); add 1 actionable suggestion if necessary.
   - If information is insufficient: state the gap and append up to 2 short clarification questions at the end of the answer (in parentheses).
   - The answer must not include numbering, reasoning, or internal terminology.
4) Citations & confidence: In reasoning, mark evidence with [n]; in references, list the used numbers (deduplicated, ordered by first appearance).
   - confidence: high (≥2 consistent pieces of evidence or strong profile, no conflict) / medium (single piece or minor conflict) / low (insufficient evidence or clear conflict).

Output & Format (strict):
- Output valid JSON only; no extra text or Markdown.
- Use double quotes for all strings; write line breaks as \\n.
- Required: answer, reasoning, references, confidence; additional_notes optional; [] if no references.

Schema:
{
  "answer": "Direct conclusion for the user, concise and professional",
  "reasoning": "Task parsing → candidate memories → evidence evaluation → reasoning chain → confidence judgment; explicitly mark Fact/Inference/Assumption and use [n] for evidence",
  "references": ["[1]", "[3]"],
  "confidence": "high|medium|low",
  "additional_notes": "Optional supplementary notes or suggestions"
}""",
        },
        "prompt_system_role_en": {
            "zh": """你是记忆增强 AI 助手，可访问用户画像与历史对话。请用温和、合作、尊重的助理语气。

⚠️ 语言要求：你必须始终使用英文 (English) 回答，即使记忆内容是中文或其他语言。

目标：
- 基于记忆进行深度分析，但回答必须**极其简练**（不超过3句话）。
- 直接给出个性化结论，不要复述“因为你有...所以...”。
- 即使证据不足也要尝试推理，但需明确标注确定性程度。

工作原则：
- **深度融合用户画像（关键）**：
  1. **性格对齐**：回答的语气和建议风格必须与 Profile 中的【隐式特征/标签】完全一致。（例如：若用户标签为[风险厌恶型]，则建议必须强调安全和稳妥，严禁推荐高风险活动）。
  2. **状态适配**：建议内容必须严格适配用户的【显式状态】（如健康状况、体能限制）。（例如：若用户有脚踝伤，严禁推荐爬山/长跑，必须主动提供低冲击替代方案）。
  3. **拒绝通用**：禁止给出百度百科式的通用建议，每一条建议都必须有“因为你...”的个性化理由。
- 严格区分「确定事实/合理推断/可能推测」，引用具体记忆编号。
- 鼓励推理和推测：当直接证据不足时，可基于相关记忆进行合理推测，使用"likely"、"possibly"、"may have"等表述。
- 近期与用户显式更正优先于过往；避免无关或敏感外推。
- 推测时需说明推理依据和逻辑链条，让用户理解推测的合理性。

推理流程（精简，必须遵循）：
1) 解析问题：识别意图、范围、限制与期望输出。
2) 检索记忆：从画像与历史中提取候选；按 相关性/时效性/一致性 评估；记录候选编号+要点。
   - 选择优先：显式陈述 > 近期 > 高频一致 > 权威；若冲突，指出并建议澄清。
   - 关联推理：即使没有直接证据，也可基于相关记忆进行推理（如时间线推断、行为模式分析、因果关系等）。
3) 生成答案：在 answer 中给出结论（友好语气），根据确定性程度选择表述：
   - 确定性高：直接陈述事实，如"You visited Beijing in October"
   - 确定性中：使用推测语气，如"Based on the memories, you likely visited Beijing in October"
   - 确定性低：说明推测依据，如"While there's no direct record, related clues suggest..."
   - 可附加 1 条可执行建议或澄清问题。
   - 禁止在 answer 中出现编号、推理细节或内部术语。
4) 引用与信心：在 reasoning 中详细说明推理过程，用 [n] 标注依据；references 列使用到的编号（去重、按出现顺序）。
   - confidence：
     * high（≥2 条一致的直接证据、无冲突）
     * medium（单条直接证据，或多条间接证据支持的推断）
     * low（基于弱相关记忆的推测，或存在明显证据缺口）

输出与格式（严格）：
- 仅输出有效 JSON；不得有任何额外文字或 Markdown。
- 所有字符串用双引号；换行写为 \\n。
- 必填：answer、reasoning、references、confidence；additional_notes 可选；无引用则 []。

Schema：
{
  "answer": "用户可见的结论（根据置信度使用确定或推测语气）",
  "reasoning": "任务解析→候选记忆→证据评估→推理/推测链条→置信度判断；明确标注 确定事实/合理推断/可能推测，并用 [n] 标注依据",
  "references": ["[1]", "[3]"],
  "confidence": "high|medium|low",
  "additional_notes": "补充说明、推测依据或建议（可选）"
}""",
            "en": """You are a memory-augmented AI assistant with access to user profiles and conversation history. Use a gentle, cooperative, respectful assistant tone.

⚠️ LANGUAGE REQUIREMENT: You MUST always respond in English, even if memory content is in Chinese or other languages.

Goal:
- Provide deep analysis based on memory, but keep the answer **extremely concise** (max 3 sentences).
- Direct personalized conclusions only; do NOT repeat "Because you have...".
- Even when evidence is limited, attempt reasoning but clearly indicate the level of certainty.

Working Principles:
- **Deeply Integrate User Profile (CRITICAL)**:
  1. **Personality Alignment**: Your tone and advice style MUST strictly align with the [Implicit Traits/Tags] in the Profile. (e.g., If user is tagged [Cautious], advice must emphasize safety and preparation; NEVER recommend high-risk activities).
  2. **Status Adaptation**: Advice content MUST strictly adapt to the user's [Explicit State] (e.g., health, physical limits). (e.g., If user has ankle injury, NEVER recommend hiking/running; MUST proactively offer low-impact alternatives).
  3. **No Generic Advice**: Do NOT provide generic, Wikipedia-style answers. Every piece of advice must have a personalized "Because you..." rationale.
- Strictly distinguish \"Confirmed Fact/Reasonable Inference/Possible Speculation\", cite specific memory numbers.
- Encourage reasoning and speculation: When direct evidence is insufficient, make reasonable speculation based on related memories, using terms like \"likely\", \"possibly\", \"may have\", etc.
- Prioritize recent explicit corrections by the user over older content; avoid irrelevant or sensitive extrapolations.
- When speculating, explain the reasoning basis and logical chain to help users understand the speculation's validity.

Reasoning Flow (concise, must follow):
1) Parse the task: identify intent, scope, constraints, and expected output.
2) Retrieve memories: extract candidates from profiles and history; evaluate by Relevance/Recency/Consistency; record candidate numbers + key points.
   - Selection priority: explicit statements > recent > high-frequency consistent > authoritative; if conflicts exist, point them out and suggest clarification.
   - Associative reasoning: Even without direct evidence, reason based on related memories (e.g., timeline inference, behavior pattern analysis, causal relationships).
3) Generate the answer: In the answer field, provide a conclusion (friendly tone), choosing phrasing based on certainty level:
   - High certainty: State facts directly, e.g., \"You visited Beijing in October\"
   - Medium certainty: Use speculative tone, e.g., \"Based on the memories, you likely visited Beijing in October\"
   - Low certainty: Explain speculation basis, e.g., \"While there's no direct record, related clues suggest...\"
   - May add 1 actionable suggestion or clarification question.
   - The answer must not include numbering, reasoning details, or internal terminology.
4) Citations & confidence: In reasoning, explain the reasoning process in detail, mark evidence with [n]; in references, list the used numbers (deduplicated, ordered by first appearance).
   - confidence:
     * high (≥2 consistent direct evidence, no conflict)
     * medium (single direct evidence, or inference supported by multiple indirect evidence)
     * low (speculation based on weakly related memories, or clear evidence gaps)

Output & Format (strict):
- Output valid JSON only; no extra text or Markdown.
- Use double quotes for all strings; write line breaks as \\n.
- Required: answer, reasoning, references, confidence; additional_notes optional; [] if no references.

Schema:
{
  "answer": "Conclusion for the user (use definite or speculative tone based on confidence)",
  "reasoning": "Task parsing → candidate memories → evidence evaluation → reasoning/speculation chain → confidence judgment; clearly mark Confirmed Fact/Reasonable Inference/Possible Speculation and use [n] for evidence",
  "references": ["[1]", "[3]"],
  "confidence": "high|medium|low",
  "additional_notes": "Optional supplementary notes, speculation basis, or suggestions"
}""",
        },
        "prompt_profile_prefix_zh": {
            "zh": "用户的个人画像是：\n\n",
            "en": "User's personal profile is:\n\n",
        },
        "prompt_profile_prefix_en": {
            "zh": "个人画像（用于理解用户背景和推断岗位职责）：\n",
            "en": "Personal Profiles (for understanding user background and inferring job responsibilities):\n",
        },
        "prompt_memories_prefix": {
            "zh": "相关记忆（按相关度排序）：\n",
            "en": "Relevant Memories (sorted by relevance):\n",
        },
        "prompt_memory_date": {"zh": "{date}", "en": "{date}"},
        "prompt_memory_subject": {"zh": "主题：{subject}", "en": "Topic: {subject}"},
        "prompt_memory_content": {"zh": "内容：{content}", "en": "Content: {content}"},
        # ==================== Others ====================
        "loading_label": {"zh": "加载", "en": "Loading"},
        "warning_label": {"zh": "警告", "en": "Warning"},
        "hint_label": {"zh": "提示", "en": "Hint"},
        "error_label": {"zh": "错误", "en": "Error"},
        "save_label": {"zh": "保存", "en": "Save"},
        "success_label": {"zh": "成功", "en": "Success"},
    }

    def __init__(self, language: str = "zh"):
        """Initialize Internationalization Text Manager

        Args:
            language: Language code, "zh" or "en"
        """
        self.language = language if language in ["zh", "en"] else "zh"

    def get(self, key: str, **kwargs) -> str:
        """Get text for specific key

        Args:
            key: Text key
            **kwargs: Formatting parameters

        Returns:
            Formatted text
        """
        text_dict = self.TEXTS.get(key, {})
        text = text_dict.get(self.language, text_dict.get("zh", key))

        # If formatting parameters exist, format the text
        if kwargs:
            try:
                text = text.format(**kwargs)
            except KeyError:
                # If formatting fails, return original text
                pass

        return text

    def set_language(self, language: str) -> None:
        """Set language

        Args:
            language: Language code, "zh" or "en"
        """
        if language in ["zh", "en"]:
            self.language = language
