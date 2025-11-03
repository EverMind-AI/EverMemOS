"""记忆增强对话脚本 - 基于历史记忆的智能对话助手

本脚本实现了一个支持记忆检索和个人画像的对话系统。

主要功能：
1. 自动列出可用的群组对话
2. 加载群组内所有用户的个人画像
3. 基于用户输入检索相关记忆
4. 结合记忆和画像生成回答
5. 维护对话历史（最近5轮）
6. 对话历史持久化（带时间戳）

支持的命令：
- exit: 退出对话（保存历史）
- clear: 清空当前对话历史
- reload: 重新加载记忆和画像
- help: 显示帮助信息

使用方法：
    python chat_with_memory.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass



# 尝试导入 readline 以支持行编辑功能

import readline
PROJECT_ROOT = Path(__file__).resolve().parents[1]
READLINE_AVAILABLE = True

from .memory_config import (
    ChatModeConfig,
    LLMConfig,
    EmbeddingConfig,
    MongoDBConfig,
    ScenarioType,
)
from .memory_utils import (
    ensure_mongo_beanie_ready,
    query_all_groups_from_mongodb,
    query_memcells_by_group_and_time,
    load_user_profiles_from_dir,
    get_user_name_from_profile,
    VectorSimilarityStrategy,
)
from .i18n_texts import I18nTexts
from memory_layer.llm.llm_provider import LLMProvider
from common_utils.datetime_utils import from_iso_format, get_now_with_timezone
from common_utils.cli_ui import CLIUI
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# 结构化响应数据类
# ============================================================================


@dataclass
class StructuredResponse:
    """LLM 结构化响应数据类"""

    answer: str  # 最终回答（用户看到的）
    reasoning: str  # 推理过程（后台记录）
    references: List[str]  # 引用的记忆编号
    confidence: str  # 置信度：high/medium/low
    additional_notes: str  # 补充说明

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "StructuredResponse":
        """从 JSON 数据创建结构化响应对象

        Args:
            json_data: LLM 返回的 JSON 数据

        Returns:
            StructuredResponse 对象
        """
        return cls(
            answer=json_data.get("answer", ""),
            reasoning=json_data.get("reasoning", ""),
            references=json_data.get("references", []),
            confidence=json_data.get("confidence", "medium"),
            additional_notes=json_data.get("additional_notes", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "answer": self.answer,
            "reasoning": self.reasoning,
            "references": self.references,
            "confidence": self.confidence,
            "additional_notes": self.additional_notes,
        }


# ============================================================================
# 语言选择器
# ============================================================================


class LanguageSelector:
    """语言选择器 - 用于选择界面语言（中文/英文）"""

    @staticmethod
    def select_language() -> str:
        """交互式选择语言

        Returns:
            语言代码："zh" 或 "en"
        """
        # 使用临时的双语标题
        print()
        print("=" * 60)
        print("  🌏  语言选择 / Language Selection")
        print("=" * 60)
        print()
        print("  [1] 中文 (Chinese)")
        print("  [2] English")
        print()

        while True:
            try:
                choice = input("请选择语言 / Please select language [1-2]: ").strip()

                if not choice:
                    continue

                index = int(choice)
                if index == 1:
                    return "zh"
                elif index == 2:
                    return "en"
                else:
                    print("❌ 请输入 1 或 2 / Please enter 1 or 2\n")

            except ValueError:
                print("❌ 请输入有效的数字 / Please enter a valid number\n")
            except KeyboardInterrupt:
                print("\n")
                return "zh"  # 默认中文


# ============================================================================
# 场景模式选择器
# ============================================================================


class ScenarioSelector:
    """场景模式选择器 - 用于选择助手模式或群聊模式"""

    @staticmethod
    def select_scenario(texts: I18nTexts) -> Optional[ScenarioType]:
        """交互式选择场景模式

        Args:
            texts: 国际化文本对象

        Returns:
            ScenarioType.ASSISTANT 或 ScenarioType.GROUP_CHAT，取消返回 None
        """
        ui = CLIUI()
        print()
        ui.section_heading(texts.get("scenario_selection_title"))
        print()

        # 显示模式列表
        print(f"  [1] {texts.get('scenario_assistant')}")
        print(f"      {texts.get('scenario_assistant_desc')}")
        print()
        print(f"  [2] {texts.get('scenario_group_chat')}")
        print(f"      {texts.get('scenario_group_chat_desc')}")
        print()

        while True:
            try:
                choice = input(f"{texts.get('scenario_prompt')}: ").strip()

                if not choice:
                    continue

                index = int(choice)
                if index == 1:
                    ui.success(
                        f"✓ {texts.get('scenario_selected')}: {texts.get('scenario_assistant')}"
                    )
                    return ScenarioType.ASSISTANT
                elif index == 2:
                    ui.success(
                        f"✓ {texts.get('scenario_selected')}: {texts.get('scenario_group_chat')}"
                    )
                    return ScenarioType.GROUP_CHAT
                else:
                    ui.error(f"✗ {texts.get('invalid_input_number')}")

            except ValueError:
                ui.error(f"✗ {texts.get('invalid_input_number')}")
            except KeyboardInterrupt:
                print("\n")
                return None


# ============================================================================
# 群组选择器
# ============================================================================


class GroupSelector:
    """群组选择器 - 用于列出和选择可用的群组对话"""

    @staticmethod
    async def list_available_groups() -> List[Dict[str, Any]]:
        """列出所有可用的群组

        Returns:
            群组列表，格式：[{"index": 1, "group_id": "xxx", "name": "xxx", "memcell_count": 76}, ...]
        """
        groups = await query_all_groups_from_mongodb()

        # 为每个群组添加索引和自定义名称
        for idx, group in enumerate(groups, start=1):
            group["index"] = idx
            # 群组自定义名称映射（可以在这里修改群组显示名称）
            group_id = group["group_id"]
            if group_id == "AI产品群":
                group["name"] = "group_chat"  # 修改群组1的显示名称
            else:
                group["name"] = group_id  # 其他群组使用原名称

        return groups

    @staticmethod
    async def select_group(
        groups: List[Dict[str, Any]], texts: I18nTexts
    ) -> Optional[str]:
        """交互式选择群组

        Args:
            groups: 群组列表
            texts: 国际化文本对象

        Returns:
            选中的 group_id，如果取消返回 None
        """
        if not groups:
            ChatUI.print_error(texts.get("groups_not_found"), texts)
            print(f"{texts.get('groups_extract_hint')}\n")
            return None

        # 显示群组列表
        ChatUI.print_group_list(groups, texts)

        # 用户输入
        while True:
            try:
                choice = input(
                    f"\n{texts.get('groups_select_prompt')} [1-{len(groups)}]: "
                ).strip()

                if not choice:
                    continue

                index = int(choice)
                if 1 <= index <= len(groups):
                    selected_group = groups[index - 1]
                    return selected_group["group_id"]
                else:
                    ChatUI.print_error(
                        texts.get("groups_select_range_error", min=1, max=len(groups)),
                        texts,
                    )

            except ValueError:
                ChatUI.print_error(texts.get("invalid_input_number"), texts)
            except KeyboardInterrupt:
                print("\n")
                ChatUI.print_info(texts.get("groups_selection_cancelled"), texts)
                return None


# ============================================================================
# 检索模式选择器
# ============================================================================


class RetrievalModeSelector:
    """检索模式选择器 - 用于选择轻量级或 Agentic 检索模式"""

    @staticmethod
    def select_retrieval_mode(texts: I18nTexts) -> Optional[str]:
        """交互式选择检索模式

        Args:
            texts: 国际化文本对象

        Returns:
            "lightweight" 或 "agentic"，取消返回 None
        """
        ui = CLIUI()
        print()
        ui.section_heading(texts.get("retrieval_mode_selection_title"))
        print()

        # 显示模式列表
        print(f"  [1] {texts.get('retrieval_mode_lightweight')}")
        print(f"      {texts.get('retrieval_mode_lightweight_desc')}")
        print()
        print(f"  [2] {texts.get('retrieval_mode_agentic')}")
        print(f"      {texts.get('retrieval_mode_agentic_desc')}")
        print()

        # 显示建议提示
        ui.note(texts.get("retrieval_mode_lightweight_note"), icon="💡")
        ui.note(texts.get("retrieval_mode_agentic_note"), icon="💡")
        print()

        while True:
            try:
                choice = input(f"{texts.get('retrieval_mode_prompt')}: ").strip()

                if not choice:
                    continue

                index = int(choice)
                if index == 1:
                    ui.success(
                        f"✓ {texts.get('retrieval_mode_selected')}: {texts.get('retrieval_mode_lightweight')}"
                    )
                    return "lightweight"
                elif index == 2:
                    ui.success(
                        f"✓ {texts.get('retrieval_mode_selected')}: {texts.get('retrieval_mode_agentic')}"
                    )
                    return "agentic"
                else:
                    ui.error(f"✗ {texts.get('invalid_input_number')}")

            except ValueError:
                ui.error(f"✗ {texts.get('invalid_input_number')}")
            except KeyboardInterrupt:
                print("\n")
                return None


# ============================================================================
# 对话会话管理
# ============================================================================


class ChatSession:
    """对话会话管理器 - 管理单个群组的对话会话"""

    def __init__(
        self,
        group_id: str,
        config: ChatModeConfig,
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
        scenario_type: ScenarioType,
        retrieval_mode: str,  # 🔥 新增：检索模式（"lightweight" 或 "agentic"）
        texts: I18nTexts,
    ):
        """初始化对话会话

        Args:
            group_id: 群组 ID
            config: 对话模式配置
            llm_config: LLM 配置
            embedding_config: 嵌入模型配置
            scenario_type: 场景类型（助手/群聊）
            retrieval_mode: 检索模式（"lightweight" 或 "agentic"）
            texts: 国际化文本对象
        """
        self.group_id = group_id
        self.config = config
        self.llm_config = llm_config
        self.embedding_config = embedding_config
        self.scenario_type = scenario_type  # 运行时场景类型
        self.retrieval_mode = retrieval_mode  # 🔥 运行时检索模式
        self.texts = texts  # 国际化文本

        # 会话状态
        self.user_profiles: Dict[str, Dict] = {}  # 所有用户的 Profile
        self.conversation_history: List[Tuple[str, str]] = (
            []
        )  # [(user_q, assistant_a), ...]
        self.memcell_count: int = 0

        # LLM 和检索策略
        self.llm_provider: Optional[LLMProvider] = None
        self.retrieval_strategy: Optional[VectorSimilarityStrategy] = None

        # 最后一次结构化响应（用于查看推理过程）
        self.last_structured_response: Optional[StructuredResponse] = None
        
        # 🔥 最后一次检索元数据（用于显示检索信息）
        self.last_retrieval_metadata: Optional[Dict[str, Any]] = None

    async def initialize(self) -> bool:
        """初始化会话（加载 Profile、创建 LLM Provider 等）

        Returns:
            初始化是否成功
        """
        try:
            # 1. 加载用户 Profile
            # 自定义群组名称映射（与 GroupSelector 保持一致）
            display_name = (
                "group_chat" if self.group_id == "AI产品群" else self.group_id
            )
            print(
                f"\n[{self.texts.get('loading_label')}] {self.texts.get('loading_group_data', name=display_name)}"
            )

            # 🔥 根据运行时的 scenario_type 和语言动态确定 Profile 路径
            # 路径格式：memcell_outputs/{scenario}_{language}/profiles[_companion]/
            scenario_name = (
                "assistant" if self.scenario_type == ScenarioType.ASSISTANT else "group_chat"
            )
            scenario_dir = self.config.memcell_output_dir / f"{scenario_name}_{self.texts.language}"
            
            # Profile 子目录：群聊用 profiles/，助手用 profiles_companion/
            if self.scenario_type == ScenarioType.ASSISTANT:
                profiles_dir = scenario_dir / "profiles_companion"
            else:
                profiles_dir = scenario_dir / "profiles"
            
            self.user_profiles = load_user_profiles_from_dir(profiles_dir)

            if not self.user_profiles:
                print(
                    f"[{self.texts.get('warning_label')}] {self.texts.get('loading_profiles_warning')}"
                )
                print(
                    f"[{self.texts.get('hint_label')}] {self.texts.get('loading_profiles_hint')}"
                )
            else:
                user_names = [
                    get_user_name_from_profile(p) or uid
                    for uid, p in self.user_profiles.items()
                ]
                print(
                    f"[{self.texts.get('loading_label')}] {self.texts.get('loading_profiles_success', count=len(self.user_profiles), names=', '.join(user_names))} ✅"
                )

            # 2. 统计 MemCell 数量（查询最近一年的数据）
            now = get_now_with_timezone()
            start_date = now - timedelta(days=self.config.time_range_days)
            memcells = await query_memcells_by_group_and_time(
                self.group_id, start_date, now
            )
            self.memcell_count = len(memcells)
            print(
                f"[{self.texts.get('loading_label')}] {self.texts.get('loading_memories_success', count=self.memcell_count)} ✅"
            )

            # 3. 加载对话历史
            loaded_history_count = await self.load_conversation_history()
            if loaded_history_count > 0:
                print(
                    f"[{self.texts.get('loading_label')}] {self.texts.get('loading_history_success', count=loaded_history_count)} ✅"
                )
            else:
                print(
                    f"[{self.texts.get('loading_label')}] {self.texts.get('loading_history_new')} ✅"
                )

            # 4. 创建 LLM Provider
            self.llm_provider = LLMProvider(
                self.llm_config.provider,
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens,
            )

            # 5. 创建检索策略
            self.retrieval_strategy = VectorSimilarityStrategy(self.embedding_config)

            print(
                f"\n[{self.texts.get('hint_label')}] {self.texts.get('loading_help_hint')}\n"
            )
            return True

        except Exception as e:
            print(
                f"\n[{self.texts.get('error_label')}] {self.texts.get('session_init_error', error=str(e))}"
            )
            import traceback

            traceback.print_exc()
            return False

    async def load_conversation_history(self) -> int:
        """从文件加载对话历史

        Returns:
            加载的对话轮数
        """
        try:
            # 自定义群组名称映射（与 GroupSelector 保持一致）
            display_name = (
                "group_chat" if self.group_id == "AI产品群" else self.group_id
            )

            # 查找最新的历史文件（支持新旧两种文件名格式）
            history_files = []
            # 查找新格式的文件名
            new_files = list(
                self.config.chat_history_dir.glob(f"{display_name}_*.json")
            )
            # 查找旧格式的文件名（为了向后兼容）
            old_files = list(
                self.config.chat_history_dir.glob(f"{self.group_id}_*.json")
            )
            # 查找更旧的格式（兼容之前的旧名称）
            very_old_files = list(
                self.config.chat_history_dir.glob("EverMem 928交付群_*.json")
            )

            history_files = sorted(
                new_files + old_files + very_old_files, reverse=True
            )  # 最新的在前

            if not history_files:
                return 0

            latest_file = history_files[0]

            with latest_file.open("r", encoding="utf-8") as fp:
                data = json.load(fp)

            # 加载最近 N 轮对话
            history = data.get("conversation_history", [])
            self.conversation_history = [
                (item["user_input"], item["assistant_response"])
                for item in history[-self.config.conversation_history_size :]
            ]

            return len(self.conversation_history)

        except Exception as e:
            print(
                f"[{self.texts.get('warning_label')}] {self.texts.get('loading_history_new')}: {e}"
            )
            return 0

    async def save_conversation_history(self) -> None:
        """保存对话历史到文件（带时间戳）"""
        try:
            # 自定义群组名称映射（与 GroupSelector 保持一致）
            display_name = (
                "group_chat" if self.group_id == "AI产品群" else self.group_id
            )
            # 文件名：group_chat_001_2025-10-24_16-30.json
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
            filename = f"{display_name}_{timestamp}.json"
            filepath = self.config.chat_history_dir / filename

            # 构建数据
            data = {
                "group_id": self.group_id,
                "last_updated": datetime.now().isoformat(),
                "conversation_history": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "user_input": user_q,
                        "assistant_response": assistant_a,
                    }
                    for user_q, assistant_a in self.conversation_history
                ],
            }

            # 保存文件
            with filepath.open("w", encoding="utf-8") as fp:
                json.dump(data, fp, ensure_ascii=False, indent=2)

            print(f"[{self.texts.get('save_label')}] {filename} ✅")

        except Exception as e:
            print(f"[{self.texts.get('error_label')}] {e}")

    def _convert_results_to_dict(self, results_tuples: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
        """将检索结果元组列表转换为标准字典格式
        
        Args:
            results_tuples: (MemCell, score) 元组列表
            
        Returns:
            标准格式的字典列表
        """
        results = []
        for mem, score in results_tuples:
            item = {
                "event_id": str(getattr(mem, "event_id", getattr(mem, "id", ""))),
                "timestamp": (
                    getattr(mem, "timestamp", None).isoformat()
                    if getattr(mem, "timestamp", None)
                    else None
                ),
                "group_id": getattr(mem, "group_id", None),
                "subject": getattr(mem, "subject", None),
                "summary": getattr(mem, "summary", None),
                "episode": getattr(mem, "episode", None),
                "participants": getattr(mem, "participants", []),
                "score": round(score, 4),
            }
            results.append(item)
        return results

    async def retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """检索相关记忆（支持多种检索模式）

        Args:
            query: 用户查询

        Returns:
            检索到的记忆列表
        """
        # 查询时间范围内的 MemCell
        now = get_now_with_timezone()
        start_date = now - timedelta(days=self.config.time_range_days)

        candidates = await query_memcells_by_group_and_time(
            self.group_id, start_date, now
        )

        if not candidates:
            self.last_retrieval_metadata = {"retrieval_mode": self.retrieval_mode, "total_latency_ms": 0.0}
            return []
        
        # 🔥 根据检索模式执行不同的检索逻辑
        if self.retrieval_mode == "lightweight":
            # 轻量级检索：Embedding + BM25 + RRF 融合
            from demo.memory_utils import lightweight_retrieval
            
            results_tuples, metadata = await lightweight_retrieval(
                query=query,
                candidates=candidates,
                embedding_config=self.embedding_config,
                emb_top_n=self.config.lightweight_emb_top_n,
                bm25_top_n=self.config.lightweight_bm25_top_n,
                final_top_n=self.config.lightweight_final_top_n
            )
            
            # 保存元数据
            self.last_retrieval_metadata = metadata
            
            # 转换为标准格式
            return self._convert_results_to_dict(results_tuples)
        
        elif self.retrieval_mode == "agentic":
            # Agentic 检索：LLM 引导的多轮检索
            from demo.memory_utils import agentic_retrieval
            
            results_tuples, metadata = await agentic_retrieval(
                query=query,
                candidates=candidates,
                embedding_config=self.embedding_config,
                llm_provider=self.llm_provider,
                config=self.config
            )
            
            # 保存元数据
            self.last_retrieval_metadata = metadata
            
            # 转换为标准格式
            return self._convert_results_to_dict(results_tuples)
        
        else:
            # 回退到默认检索（保持向后兼容）
            results = await self.retrieval_strategy.retrieve(
                query=query, candidates=candidates, top_k=self.config.top_k_memories
            )

            if self.scenario_type == ScenarioType.ASSISTANT:
                results_semantic = await self.retrieval_strategy.retrieve_semantic(
                    query=query,
                    candidates=candidates,
                    date_query=datetime.strptime("2024-10-27", "%Y-%m-%d"),
                    top_k=self.config.top_k_memories,
                )
                results = results + results_semantic
            
            self.last_retrieval_metadata = {"retrieval_mode": "default", "total_latency_ms": 0.0}
            return results

    def build_prompt(
        self, user_query: str, memories: List[Dict[str, Any]], profiles: Dict[str, Dict]
    ) -> List[Dict[str, str]]:
        """构建 Chat Messages 格式的 Prompt

        Args:
            user_query: 用户查询
            memories: 检索到的记忆列表
            profiles: 所有用户的 Profile

        Returns:
            Chat Messages 列表
        """
        messages = []

        # 1. System Message - 角色定义和 JSON 输出要求（根据语言选择）
        lang_key = "zh" if self.texts.language == "zh" else "en"
        system_content = self.texts.get(f"prompt_system_role_{lang_key}")

        messages.append({"role": "system", "content": system_content})

        # 2. Profile Context - 个人画像（所有用户）

        if profiles:
            if self.scenario_type == ScenarioType.ASSISTANT:
                profile_data = profiles.get("user_001")
                if not profile_data:
                    # 回退：优先选择非 assistant 的用户画像
                    for _uid, _pf in profiles.items():
                        if (_pf or {}).get("user_name") != "assistant":
                            profile_data = _pf
                            break
                if profile_data:
                    profile_content = self._sanitize_companion_profile_for_prompt(
                        profile_data
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": self.texts.get(
                                f"prompt_profile_prefix_{lang_key}"
                            )
                            + profile_content,
                        }
                    )

            elif self.scenario_type == ScenarioType.GROUP_CHAT:
                # print(profiles)
                profile_list = []
                for user_id, profile in profiles.items():
                    user_name = get_user_name_from_profile(profile) or user_id
                    # print(user_name)
                    user_profile = profiles[user_id]
                    profile_list.append(
                        json.dumps(user_profile, ensure_ascii=False, indent=2)
                    )

                if profile_list != []:
                    profile_content = self.texts.get(
                        f"prompt_profile_prefix_{lang_key}"
                    ) + "\n".join(profile_list)
                    # print(profile_content)
                    messages.append({"role": "system", "content": profile_content})

        # 3. Retrieved Memories - 检索到的记忆
        if memories:
            memory_lines = []
            for i, mem in enumerate(memories, start=1):
                timestamp = mem.get("timestamp", "")[:10]  # 只取日期部分
                subject = mem.get("subject", "")
                summary = mem.get("summary", "")
                episode = mem.get("episode", "")
                # 详细版：提供更多上下文信息以便推理
                # 格式：[序号] 日期 | 主题：xxx | 内容：xxx
                parts = [
                    f"[{i}] {self.texts.get('prompt_memory_date', date=timestamp)}"
                ]
                if subject:
                    parts.append(
                        self.texts.get("prompt_memory_subject", subject=subject)
                    )
                if summary:
                    parts.append(
                        self.texts.get("prompt_memory_content", content=summary)
                    )
                if episode:
                    parts.append(
                        self.texts.get("prompt_memory_episode", episode=episode)
                    )
                memory_lines.append(" | ".join(parts))

            memory_content = self.texts.get("prompt_memories_prefix") + "\n".join(
                memory_lines
            )
            messages.append({"role": "system", "content": memory_content})

        # 4. Conversation History - 对话历史（最近 N 轮）
        for user_q, assistant_a in self.conversation_history[
            -self.config.conversation_history_size :
        ]:
            messages.append({"role": "user", "content": user_q})
            messages.append({"role": "assistant", "content": assistant_a})

        # 5. Current Question - 当前问题
        messages.append({"role": "user", "content": user_query})

        return messages

    def _sanitize_companion_profile_for_prompt(self, profile: Dict[str, Any]) -> str:
        """生成用于 Prompt 的精简 JSON（去掉 null、去掉 reasoning、去掉所有 evidence* 相关字段）。"""

        def is_evidence_key(key: str) -> bool:
            try:
                return "evidence" in key.lower()
            except Exception:
                return False

        def cleanse(obj: Any) -> Any:
            if isinstance(obj, dict):
                cleaned: Dict[str, Any] = {}
                for k, v in obj.items():
                    # 移除推理字段与空值
                    if k in ("output_reasoning", "reasoning", "outputReasoning"):
                        continue
                    if v is None:
                        continue
                    cv = cleanse(v)
                    if cv is not None:
                        cleaned[k] = cv
                return cleaned
            if isinstance(obj, list):
                new_list: List[Any] = []
                for item in obj:
                    if item is None:
                        continue
                    if isinstance(item, dict):
                        filtered = {
                            kk: vv
                            for kk, vv in item.items()
                            if (not is_evidence_key(kk)) and vv is not None
                        }
                        cleaned_item = cleanse(filtered)
                        if cleaned_item:
                            new_list.append(cleaned_item)
                    else:
                        new_list.append(item)
                return new_list
            return obj

        sanitized = cleanse(profile)
        try:
            return json.dumps(sanitized, ensure_ascii=False, indent=2)
        except Exception:
            return json.dumps(profile, ensure_ascii=False)

    async def chat(self, user_input: str) -> str:
        """核心对话逻辑

        Args:
            user_input: 用户输入

        Returns:
            助手回答
        """
        # 1. 检索记忆（新方法返回单个列表）
        memories = await self.retrieve_memories(user_input)

        # 2. 显示检索结果（如果配置启用）- 只显示前 5 条
        if self.config.show_retrieved_memories:
            if memories:
                # 🔥 显示检索模式和耗时信息
                retrieval_mode_display = self.retrieval_mode
                ChatUI.print_retrieved_memories(
                    memories[:5],
                    total_count=len(memories),
                    texts=self.texts,
                    retrieval_metadata=self.last_retrieval_metadata,  # 🔥 传递元数据
                )
        
        # 3. 构建 Prompt（使用全部记忆）
        messages = self.build_prompt(user_input, memories, self.user_profiles)

        # 4. 显示生成进度提示
        ChatUI.print_generating_indicator(self.texts)

        # 5. 调用 LLM
        try:
            # 将 messages 格式转换为单个 prompt（LLMProvider.generate 需要字符串）
            # 或者直接使用底层 provider 的 chat 功能
            if hasattr(self.llm_provider, 'provider') and hasattr(
                self.llm_provider.provider, 'chat_with_messages'
            ):
                # 如果 provider 支持直接使用 messages
                raw_response = await self.llm_provider.provider.chat_with_messages(
                    messages
                )
            else:
                # 回退方案：将 messages 转换为单个 prompt
                prompt_parts = []
                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        prompt_parts.append(f"System: {content}")
                    elif role == "user":
                        prompt_parts.append(f"User: {content}")
                    elif role == "assistant":
                        prompt_parts.append(f"Assistant: {content}")

                prompt = "\n\n".join(prompt_parts)
                raw_response = await self.llm_provider.generate(prompt)

            raw_response = raw_response.strip()

            # 6. 清除生成进度提示，显示完成标识
            ChatUI.print_generation_complete(self.texts)

            # 7. 解析 JSON 响应
            structured_response = self.parse_llm_response(raw_response)

            if structured_response is None:
                # JSON 解析失败，使用回退方案
                print(
                    f"[{self.texts.get('warning_label')}] LLM 输出非 JSON 格式，使用回退方案"
                )
                structured_response = self.create_fallback_response(raw_response)

            # 8. 保存结构化响应
            self.last_structured_response = structured_response

            # 9. 显示结构化信息（元数据）
            if self.config.show_reasoning_metadata:
                ChatUI.print_response_metadata(structured_response, self.texts)

            # 10. 提取用户看到的回答
            assistant_response = structured_response.answer

        except Exception as e:
            # 清除进度提示
            ChatUI.clear_progress_indicator()
            error_msg = f"[{self.texts.get('error_label')}] {self.texts.get('chat_llm_error', error=str(e))}"
            print(f"\n{error_msg}")
            import traceback

            traceback.print_exc()
            return error_msg

        # 11. 更新对话历史（存储完整的 answer）
        self.conversation_history.append((user_input, assistant_response))

        # 12. 保持历史窗口大小
        if len(self.conversation_history) > self.config.conversation_history_size:
            self.conversation_history = self.conversation_history[
                -self.config.conversation_history_size :
            ]

        return assistant_response

    def parse_llm_response(self, raw_response: str) -> Optional[StructuredResponse]:
        """解析 LLM 的 JSON 响应

        Args:
            raw_response: LLM 原始响应文本

        Returns:
            StructuredResponse 对象，解析失败返回 None
        """
        try:
            # 尝试直接解析 JSON
            json_data = json.loads(raw_response.strip())
            return StructuredResponse.from_json(json_data)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取 JSON 部分
            try:
                # 查找 JSON 代码块（```json ... ```）
                import re

                json_match = re.search(
                    r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(1)
                    json_data = json.loads(json_str)
                    return StructuredResponse.from_json(json_data)

                # 查找普通代码块（``` ... ```）
                code_match = re.search(
                    r'```\s*(\{.*?\})\s*```', raw_response, re.DOTALL
                )
                if code_match:
                    json_str = code_match.group(1)
                    json_data = json.loads(json_str)
                    return StructuredResponse.from_json(json_data)

                # 尝试查找 { ... } 结构
                brace_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if brace_match:
                    json_str = brace_match.group(0)
                    json_data = json.loads(json_str)
                    return StructuredResponse.from_json(json_data)

                return None
            except Exception as e:
                print(f"[警告] JSON 提取失败: {e}")
                return None
        except Exception as e:
            print(f"[警告] 响应解析异常: {e}")
            return None

    def create_fallback_response(self, raw_text: str) -> StructuredResponse:
        """创建回退响应（当 JSON 解析失败时）

        Args:
            raw_text: 原始响应文本

        Returns:
            StructuredResponse 对象
        """
        return StructuredResponse(
            answer=raw_text,
            reasoning="LLM 未按 JSON 格式输出，无法提取推理过程",
            references=[],
            confidence="low",
            additional_notes="注意：此回答未经结构化处理",
        )

    def show_reasoning(self) -> None:
        """显示最后一次回答的完整推理过程"""
        if self.last_structured_response is None:
            ChatUI.print_info(self.texts.get("cmd_reasoning_no_data"), self.texts)
            return

        ChatUI.print_full_reasoning(self.last_structured_response, self.texts)

    def clear_history(self) -> None:
        """清空对话历史"""
        count = len(self.conversation_history)
        self.conversation_history = []
        ChatUI.print_info(self.texts.get("cmd_clear_done", count=count), self.texts)

    async def reload_data(self) -> None:
        """重新加载记忆和画像"""
        # 自定义群组名称映射（与 GroupSelector 保持一致）
        display_name = "group_chat" if self.group_id == "AI产品群" else self.group_id

        ui = ChatUI._ui()
        print()
        ui.note(self.texts.get("cmd_reload_refreshing", name=display_name), icon="🔄")

        # 🔥 根据运行时的 scenario_type 和语言动态确定 Profile 路径
        scenario_name = (
            "assistant" if self.scenario_type == ScenarioType.ASSISTANT else "group_chat"
        )
        scenario_dir = self.config.memcell_output_dir / f"{scenario_name}_{self.texts.language}"
        
        if self.scenario_type == ScenarioType.ASSISTANT:
            profiles_dir = scenario_dir / "profiles_companion"
        else:
            profiles_dir = scenario_dir / "profiles"
        
        self.user_profiles = load_user_profiles_from_dir(profiles_dir)

        # 重新统计 MemCell 数量
        now = get_now_with_timezone()
        start_date = now - timedelta(days=self.config.time_range_days)
        memcells = await query_memcells_by_group_and_time(
            self.group_id, start_date, now
        )
        self.memcell_count = len(memcells)

        print()
        ui.success(
            f"✓ {self.texts.get('cmd_reload_complete', users=len(self.user_profiles), memories=self.memcell_count)}"
        )
        print()


# ============================================================================
# 美观的终端界面
# ============================================================================


class ChatUI:
    """终端界面工具类 - 提供美观的输出格式"""

    @staticmethod
    def _ui() -> CLIUI:
        """获取一个宽度自适应、可选彩色的 UI 实例。

        颜色开关遵循环境变量：
        - NO_COLOR: 任意值表示禁用颜色
        - CLI_UI_COLOR=0 禁用颜色；其他值或未设置则根据 TTY 自动启用
        """
        return CLIUI()

    @staticmethod
    def clear_screen():
        """清空屏幕内容

        使用 ANSI 转义序列清屏并将光标移动到左上角，
        兼容 Unix/Linux/macOS 和 Windows 终端。
        """
        import os

        # 优先使用 ANSI 转义序列（更通用）
        print("\033[2J\033[H", end="")
        # 刷新输出缓冲区
        import sys

        sys.stdout.flush()

    @staticmethod
    def print_banner(texts: I18nTexts):
        """打印欢迎横幅

        Args:
            texts: 国际化文本对象
        """
        ui = ChatUI._ui()
        print()
        ui.banner(texts.get("banner_title"), subtitle=texts.get("banner_subtitle"))
        if READLINE_AVAILABLE:
            ui.note(texts.get("readline_available"), icon="💡")
        else:
            ui.note(texts.get("readline_unavailable"), icon="💡")
        print()

    @staticmethod
    def print_group_list(groups: List[Dict[str, Any]], texts: I18nTexts):
        """显示群组列表（表格格式）

        Args:
            groups: 群组列表
            texts: 国际化文本对象
        """
        ui = ChatUI._ui()
        print()
        ui.section_heading(texts.get("groups_available_title"))

        rows = []
        for group in groups:
            index = group["index"]
            group_id = group["group_id"]
            name = group.get("name", group_id)
            count = group["memcell_count"]
            count_text = f"💾 {count} " + (
                "memories" if texts.language == "en" else "条记忆"
            )
            rows.append([f"[{index}]", group_id, f"📝 \"{name}\"", count_text])

        headers = [
            texts.get("table_header_index"),
            texts.get("table_header_group"),
            texts.get("table_header_name"),
            texts.get("table_header_count"),
        ]
        ui.table(headers=headers, rows=rows, aligns=["right", "left", "left", "right"])

    @staticmethod
    def print_retrieved_memories(
        memories: List[Dict[str, Any]],
        total_count: Optional[int],
        texts: I18nTexts,
        retrieval_metadata: Optional[Dict[str, Any]] = None,
    ):
        """显示检索到的记忆（简洁版）

        Args:
            memories: 记忆列表（显示用）
            total_count: 实际检索到的总数（保留参数以兼容旧代码，但不再使用）
            texts: 国际化文本对象
            retrieval_metadata: 检索元数据（可选）
        """
        ui = ChatUI._ui()

        # 🔥 简化标题：显示检索完成和显示数量
        heading = f"🔍 {texts.get('retrieval_complete')}"
        shown_count = len(memories)
        if shown_count > 0:
            heading += f" - {texts.get('retrieval_showing', shown=shown_count)}"
        
        # 🔥 如果有元数据，显示检索模式和耗时
        if retrieval_metadata:
            retrieval_mode = retrieval_metadata.get("retrieval_mode", "default")
            latency_ms = retrieval_metadata.get("total_latency_ms", 0.0)
            is_multi_round = retrieval_metadata.get("is_multi_round", False)
            
            if retrieval_mode == "lightweight":
                mode_text = texts.get("retrieval_mode_lightweight")
            elif retrieval_mode == "agentic":
                mode_text = texts.get("retrieval_mode_agentic")
                if is_multi_round:
                    mode_text += f" ({texts.get('retrieval_multi_round')})"
                else:
                    mode_text += f" ({texts.get('retrieval_single_round')})"
            else:
                mode_text = "Default"
            
            heading += f" | {mode_text} | {texts.get('retrieval_latency', latency=int(latency_ms))}"

        ui.section_heading(heading)

        lines = []
        for i, mem in enumerate(memories, start=1):
            timestamp = mem.get("timestamp", "")[:10]
            subject = mem.get("subject", "")
            summary = mem.get("summary", "")
            content = subject or summary or ""
            lines.append(f"📌 [{i:2d}]  {timestamp}  │  {content}")
        if lines:
            ui.panel(lines)

    @staticmethod
    def print_generating_indicator(texts: I18nTexts):
        """显示生成进度提示

        Args:
            texts: 国际化文本对象
        """
        ui = ChatUI._ui()
        print()
        ui.note(f"🤔 {texts.get('chat_generating')}", icon="⏳")

    @staticmethod
    def print_generation_complete(texts: I18nTexts):
        """清除生成提示并显示完成标识

        Args:
            texts: 国际化文本对象
        """
        # 使用 ANSI 转义序列清除上一行并移动光标
        print("\r\033[K", end="")  # 清除当前行
        print("\033[A\033[K", end="")  # 向上移动一行并清除
        print("\033[A\033[K", end="")  # 再向上移动一行并清除（处理两行的提示）
        ui = ChatUI._ui()
        ui.success(f"✓ {texts.get('chat_generation_complete')}")

    @staticmethod
    def clear_progress_indicator():
        """清除进度提示（出错时使用）"""
        # 清除进度提示行
        print("\r\033[K", end="")
        print("\033[A\033[K", end="")
        print("\033[A\033[K", end="")

    @staticmethod
    def print_response_metadata(response: "StructuredResponse", texts: I18nTexts):
        """显示结构化响应的元数据

        Args:
            response: 结构化响应对象
            texts: 国际化文本对象
        """
        ui = ChatUI._ui()
        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        emoji = confidence_emoji.get(response.confidence, "⚪")
        refs = (
            f"📎 {texts.get('response_references')}: {', '.join(response.references)}"
            if response.references
            else ""
        )
        line = (
            f"{emoji} {texts.get('response_confidence')}: {response.confidence.upper()}"
        )
        if refs:
            line += f"  |  {refs}"
        print()
        ui.text(line)

    @staticmethod
    def print_assistant_response(response: str, texts: I18nTexts):
        """显示助手回答

        Args:
            response: 助手回答文本
            texts: 国际化文本对象
        """
        ui = ChatUI._ui()
        print()  # 额外的空行，增加可读性
        ui.panel([response], title=texts.get("response_assistant_title"))
        ui.rule()
        print()  # 在回答后留出空间

    @staticmethod
    def print_full_reasoning(response: "StructuredResponse", texts: I18nTexts):
        """显示完整的推理过程

        Args:
            response: 结构化响应对象
            texts: 国际化文本对象
        """
        ui = ChatUI._ui()
        print()
        ui.section_heading(texts.get("response_reasoning_title"))
        ui.panel([texts.get("response_answer_label")])
        ui.panel(response.answer.split("\n"))
        ui.panel([texts.get("response_reasoning_label")])
        ui.panel(response.reasoning.split("\n"))

        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
        emoji = confidence_emoji.get(response.confidence, "⚪")
        refs_text = (
            ', '.join(response.references)
            if response.references
            else texts.get("response_no_references")
        )
        meta_lines = [
            f"{emoji} {texts.get('response_confidence')}: {response.confidence.upper()}",
            f"📎 {texts.get('response_references')}: {refs_text}",
        ]
        ui.panel([texts.get("response_metadata_label")])
        ui.panel(meta_lines)

        if response.additional_notes:
            ui.panel([texts.get("response_notes_label")])
            ui.panel(response.additional_notes.split("\n"))
        print()

    @staticmethod
    def print_help(texts: I18nTexts):
        """显示帮助信息

        Args:
            texts: 国际化文本对象
        """
        ui = ChatUI._ui()
        print()
        ui.section_heading(texts.get("cmd_help_title"))
        lines = [
            f"🚪  {texts.get('cmd_exit')}",
            f"🧹  {texts.get('cmd_clear')}",
            f"🔄  {texts.get('cmd_reload')}",
            f"🧠  {texts.get('cmd_reasoning')}",
            f"❓  {texts.get('cmd_help')}",
        ]
        ui.panel(lines)
        print()

    @staticmethod
    def print_info(message: str, texts: I18nTexts):
        """显示信息提示

        Args:
            message: 提示信息
            texts: 国际化文本对象
        """
        ui = ChatUI._ui()
        print()
        ui.success(f"✓ {message}")
        print()

    @staticmethod
    def print_error(message: str, texts: I18nTexts):
        """显示错误信息

        Args:
            message: 错误信息
            texts: 国际化文本对象
        """
        ui = ChatUI._ui()
        print()
        ui.error(f"✗ {message}")
        print()


# ============================================================================
# 主入口函数
# ============================================================================


async def main():
    """主入口函数 - 启动对话系统"""

    # 0. 配置日志级别 - 隐藏技术日志，只显示警告和错误
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    # 确保所有第三方库的日志也遵循此设置
    logging.getLogger().setLevel(logging.WARNING)

    # 1. 启用行编辑功能
    if READLINE_AVAILABLE:
        # 设置历史记录文件
        history_file = PROJECT_ROOT / "demo" / ".chat_history"
        try:
            if history_file.exists():
                readline.read_history_file(str(history_file))
            # 设置历史记录最大长度
            readline.set_history_length(1000)
        except Exception:
            pass  # 忽略历史记录加载错误

    # 2. 语言选择
    language = LanguageSelector.select_language()
    texts = I18nTexts(language)

    # 3. 清屏，准备显示欢迎界面
    ChatUI.clear_screen()

    # 4. 打印欢迎横幅
    ChatUI.print_banner(texts)

    # 5. 场景模式选择
    scenario_type = ScenarioSelector.select_scenario(texts)
    if not scenario_type:
        ChatUI.print_info(texts.get("groups_not_selected_exit"), texts)
        return

    # 6. 刷新屏幕，保留横幅
    ChatUI.clear_screen()
    ChatUI.print_banner(texts)

    # 7. 初始化配置
    chat_config = ChatModeConfig()
    llm_config = LLMConfig()
    embedding_config = EmbeddingConfig()
    mongo_config = MongoDBConfig()

    # 8. 验证 LLM API Key
    import os

    api_key_present = any(
        [
            llm_config.api_key,
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
        ]
    )
    if not api_key_present:
        ChatUI.print_error(texts.get("config_api_key_missing"), texts)
        print(f"{texts.get('config_api_key_hint')}\n")
        return

    # 9. 初始化 MongoDB
    ui = ChatUI._ui()
    ui.note(texts.get("mongodb_connecting"), icon="🔌")
    try:
        await ensure_mongo_beanie_ready(mongo_config)
        print("\r\033[K", end="")  # 清除 "连接 MongoDB..." 行
        print("\033[A\033[K", end="")
    except Exception as e:
        ChatUI.print_error(texts.get("mongodb_init_failed", error=str(e)), texts)
        return

    # 10. 群组选择
    groups = await GroupSelector.list_available_groups()
    selected_group_id = await GroupSelector.select_group(groups, texts)

    if not selected_group_id:
        ChatUI.print_info(texts.get("groups_not_selected_exit"), texts)
        return

    # 11. 🔥 检索模式选择
    retrieval_mode = RetrievalModeSelector.select_retrieval_mode(texts)
    
    if not retrieval_mode:
        ChatUI.print_info(texts.get("groups_not_selected_exit"), texts)
        return

    # 12. 刷新屏幕，保留横幅
    ChatUI.clear_screen()
    ChatUI.print_banner(texts)

    # 13. 创建并初始化对话会话
    session = ChatSession(
        group_id=selected_group_id,
        config=chat_config,
        llm_config=llm_config,
        embedding_config=embedding_config,
        scenario_type=scenario_type,
        retrieval_mode=retrieval_mode,  # 🔥 传递检索模式
        texts=texts,
    )

    if not await session.initialize():
        ChatUI.print_error(texts.get("session_init_failed"), texts)
        return

    # 14. 进入对话循环
    ui = ChatUI._ui()
    print()
    ui.rule()
    ui.note(texts.get("chat_start_note"), icon="💬")
    ui.rule()
    print()

    while True:
        try:
            # 获取用户输入
            user_input = input(texts.get("chat_input_prompt")).strip()

            if not user_input:
                continue

            # 处理命令（命令不会被记录到对话历史）
            command = user_input.lower()

            if command == "exit":
                ui = ChatUI._ui()
                print()
                ui.note(texts.get("cmd_exit_saving"), icon="💾")
                await session.save_conversation_history()
                print()
                ui.success(f"✓ {texts.get('cmd_exit_complete')}")
                print()
                break

            elif command == "clear":
                session.clear_history()
                continue

            elif command == "reload":
                await session.reload_data()
                continue

            elif command == "reasoning":
                session.show_reasoning()
                continue

            elif command == "help":
                ChatUI.print_help(texts)
                continue

            # 执行对话（只有非命令的输入才会被记录到历史）
            response = await session.chat(user_input)
            ChatUI.print_assistant_response(response, texts)

        except KeyboardInterrupt:
            ui = ChatUI._ui()
            print("\n")
            ui.note(texts.get("cmd_interrupt_saving"), icon="⚠️")
            await session.save_conversation_history()
            print()
            ui.success(f"✓ {texts.get('cmd_exit_complete')}")
            print()
            break

        except Exception as e:
            ChatUI.print_error(texts.get("chat_error", error=str(e)), texts)
            import traceback

            traceback.print_exc()
            print()  # 额外空行，增加可读性

    # 保存输入历史
    if READLINE_AVAILABLE:
        try:
            history_file = PROJECT_ROOT / "demo" / ".chat_history"
            readline.write_history_file(str(history_file))
        except Exception:
            pass  # 忽略历史记录保存错误


if __name__ == "__main__":
    asyncio.run(main())
