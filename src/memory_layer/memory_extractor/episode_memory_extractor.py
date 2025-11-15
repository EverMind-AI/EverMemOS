"""
Simple Memory Extraction Base Class for EverMemOS

This module provides a simple base class for extracting memories
from boundary detection results (BoundaryResult).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import re, json, asyncio, uuid


# 使用动态语言提示词导入（根据 MEMORY_LANGUAGE 环境变量自动选择）
from ..prompts import (
    EPISODE_GENERATION_PROMPT,
    GROUP_EPISODE_GENERATION_PROMPT,
    DEFAULT_CUSTOM_INSTRUCTIONS,
)

# 评估专用提示词
from ..prompts.eval.episode_mem_prompts import (
    EPISODE_GENERATION_PROMPT as EVAL_EPISODE_GENERATION_PROMPT,
    GROUP_EPISODE_GENERATION_PROMPT as EVAL_GROUP_EPISODE_GENERATION_PROMPT,
    DEFAULT_CUSTOM_INSTRUCTIONS as EVAL_DEFAULT_CUSTOM_INSTRUCTIONS,
)


from ..llm.llm_provider import LLMProvider

from .base_memory_extractor import MemoryExtractor, MemoryExtractRequest
from .semantic_memory_extractor import SemanticMemoryExtractor
from ..types import MemoryType, Memory, RawDataType, MemCell

from common_utils.datetime_utils import get_now_with_timezone

from core.observation.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EpisodeMemory(Memory):
    """
    Simple result class for memory extraction.

    Contains the essential information for extracted memories.
    """

    event_id: str = field(default=None)

    def __post_init__(self):
        """Set memory_type to EPISODE_SUMMARY and call parent __post_init__."""
        self.memory_type = MemoryType.EPISODE_SUMMARY
        super().__post_init__()


@dataclass
class EpisodeMemoryExtractRequest(MemoryExtractRequest):
    pass


class EpisodeMemoryExtractor(MemoryExtractor):
    def __init__(self, llm_provider: LLMProvider | None = None, use_eval_prompts: bool = False):
        super().__init__(MemoryType.EPISODE_SUMMARY)
        self.llm_provider = llm_provider
        self.semantic_extractor = SemanticMemoryExtractor(self.llm_provider)
        self.use_eval_prompts = use_eval_prompts
        if self.use_eval_prompts:
            self.episode_generation_prompt = EVAL_EPISODE_GENERATION_PROMPT
            self.group_episode_generation_prompt = EVAL_GROUP_EPISODE_GENERATION_PROMPT
            self.default_custom_instructions = EVAL_DEFAULT_CUSTOM_INSTRUCTIONS
        else:
            self.episode_generation_prompt = EPISODE_GENERATION_PROMPT
            self.group_episode_generation_prompt = GROUP_EPISODE_GENERATION_PROMPT
            self.default_custom_instructions = DEFAULT_CUSTOM_INSTRUCTIONS

    def _parse_timestamp(self, timestamp) -> datetime:
        """
        解析时间戳为 datetime 对象
        支持多种格式：数字时间戳、ISO格式字符串、数字字符串等
        """
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            # Handle string timestamps (could be ISO format or timestamp string)
            try:
                if timestamp.isdigit():
                    return datetime.fromtimestamp(int(timestamp))
                else:
                    # Try parsing as ISO format
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # Fallback to current time if parsing fails
                logger.error(f"解析时间戳失败: {timestamp}")
                return get_now_with_timezone()
        else:
            # Unknown format, fallback to current time
            logger.error(f"解析时间戳失败: {timestamp}")
            return get_now_with_timezone()

    def _format_timestamp(self, dt: datetime) -> str:
        """
        格式化 datetime 为易读的字符串格式
        """
        weekday = dt.strftime("%A")  # Monday, Tuesday, etc.
        month_day = dt.strftime("%B %d, %Y")  # March 14, 2024
        time_of_day = dt.strftime("%I:%M %p")  # 3:00 PM
        return f"{month_day} ({weekday}) at {time_of_day} UTC"

    def get_conversation_text(self, data_list):
        lines = []
        for data in data_list:
            # Handle both RawData objects and dict objects
            if hasattr(data, 'content'):
                # RawData object
                speaker = data.content.get('speaker_name') or data.content.get(
                    'sender', 'Unknown'
                )
                content = data.content['content']
                timestamp = data.content['timestamp']
            else:
                # Dict object
                speaker = data.get('speaker_name') or data.get('sender', 'Unknown')
                content = data['content']
                timestamp = data['timestamp']

            if timestamp:
                lines.append(f"[{timestamp}] {speaker}: {content}")
            else:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def get_conversation_json_text(self, data_list):
        lines = []
        for data in data_list:
            # Handle both RawData objects and dict objects
            if hasattr(data, 'content'):
                # RawData object
                speaker = data.content.get('speaker_name') or data.content.get(
                    'sender', 'Unknown'
                )
                content = data.content['content']
                timestamp = data.content['timestamp']
            else:
                # Dict object
                speaker = data.get('speaker_name') or data.get('sender', 'Unknown')
                content = data['content']
                timestamp = data['timestamp']

            if timestamp:
                lines.append(
                    f"""
                {{
                    "timestamp": {timestamp},
                    "speaker": {speaker},
                    "content": {content}
                }}"""
                )
            else:
                lines.append(
                    f"""
                {{
                    "speaker": {speaker},
                    "content": {content}
                }}"""
                )
        return "\n".join(lines)

    def get_speaker_name_map(self, data_list: List[Dict[str, Any]]) -> Dict[str, str]:
        speaker_name_map = {}
        for data in data_list:
            if hasattr(data, 'content'):
                speaker_name_map[data.content.get('speaker_id')] = data.content.get(
                    'speaker_name'
                )
            else:
                speaker_name_map[data.get('speaker_id')] = data.get('speaker_name')
        return speaker_name_map

    def _extract_participant_name_map(
        self, chat_raw_data_list: List[Dict[str, Any]]
    ) -> List[str]:
        participant_name_map = {}
        for raw_data in chat_raw_data_list:
            if 'speaker_name' in raw_data and raw_data['speaker_name']:
                participant_name_map[raw_data['speaker_id']] = raw_data['speaker_name']
            if 'referList' in raw_data and raw_data['referList']:
                for refer_item in raw_data['referList']:
                    if isinstance(refer_item, dict):
                        if 'name' in refer_item and refer_item['_id']:
                            participant_name_map[refer_item['_id']] = refer_item['name']
        return participant_name_map

    async def _trigger_semantic_extraction_async(
        self,
        episode_memories: List[EpisodeMemory],
        request: EpisodeMemoryExtractRequest,
    ):
        """
        异步触发语义记忆提取，不影响主流程
        使用并发方式处理多个episode的语义记忆提取
        """
        if not self.semantic_extractor:
            logger.debug("语义记忆提取器未初始化，跳过语义记忆提取")
            return

        logger.info("异步触发语义记忆提取...")

        # 定义单个episode的语义记忆提取函数
        async def extract_semantic_for_episode(episode_memory: EpisodeMemory):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.debug(
                        f"🧠 自动触发语义记忆提取开始: episode_memory='{episode_memory.subject}' (尝试 {attempt + 1}/{max_retries})"
                    )
                    semantic_memories = await self.semantic_extractor.generate_semantic_memories_for_episode(
                        episode_memory
                    )
                    episode_memory.semantic_memories = semantic_memories
                    logger.info(
                        f"✅ 为情景记忆 '{episode_memory.subject}' 生成了 {len(semantic_memories)} 条语义记忆"
                    )
                    return True  # 成功
                except Exception as e:
                    logger.error(
                        f"❌ 为情景记忆 '{episode_memory.subject}' 生成语义记忆时出错: {e} (尝试 {attempt + 1}/{max_retries})"
                    )

                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)
                    else:
                        logger.error(
                            f"❌ 所有重试次数均失败，未能为情景记忆 '{episode_memory.subject}' 提取语义记忆"
                        )
                        return False  # 失败
            return False

        # 并发处理所有episode的语义记忆提取
        tasks = [
            extract_semantic_for_episode(episode_memory)
            for episode_memory in episode_memories
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 统计结果
        success_count = sum(1 for result in results if result is True)
        logger.info(
            f"语义记忆提取完成: {success_count}/{len(episode_memories)} 个episode成功"
        )

    async def _trigger_semantic_extraction_for_memcell_async(
        self, memcell: MemCell, request: EpisodeMemoryExtractRequest
    ):
        """
        异步为MemCell触发语义记忆提取，不影响主流程
        """
        if not self.semantic_extractor:
            logger.debug("语义记忆提取器未初始化，跳过语义记忆提取")
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(
                    f"🧠 自动触发语义记忆提取开始: memcell='{memcell.subject}' (尝试 {attempt + 1}/{max_retries})"
                )
                semantic_memories = await self.semantic_extractor.generate_semantic_memories_for_memcell(
                    memcell
                )
                memcell.semantic_memories = semantic_memories
                logger.info(
                    f"✅ 为MemCell '{memcell.subject}' 生成了 {len(semantic_memories)} 条语义记忆"
                )
                break  # 成功则跳出重试循环
            except Exception as e:
                logger.error(
                    f"❌ 为MemCell '{memcell.subject}' 生成语义记忆时出错: {e} (尝试 {attempt + 1}/{max_retries})"
                )

                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                else:
                    logger.error(
                        f"❌ 所有重试次数均失败，未能为MemCell '{memcell.subject}' 提取语义记忆"
                    )

    async def extract_memory(
        self,
        request: EpisodeMemoryExtractRequest,
        use_group_prompt: bool = False,
        use_semantic_extraction: bool = False,
    ) -> Optional[List[EpisodeMemory]] | Optional[MemCell]:
        logger.debug(f"📚 自动触发情景记忆提取...")

        if not request.memcell_list:
            return None

        # 获取第一个 memcell 来判断类型
        first_memcell = request.memcell_list[0]

        # 根据类型选择不同的处理方式
        if first_memcell.type == RawDataType.CONVERSATION:
            all_content_text = []
            prompt_template = ""
            # 对话类型处理
            for memcell in request.memcell_list:
                # conversation_text = self.get_conversation_text(memcell.original_data)
                conversation_text = self.get_conversation_json_text(
                    memcell.original_data
                )
                all_content_text.append(conversation_text)

            # 根据使用场景选择提示词
            if use_group_prompt:
                # 与 extract_memcell 配套使用
                prompt_template = self.group_episode_generation_prompt
                content_key = "conversation"
                time_key = "conversation_start_time"
            else:
                # 单独使用
                prompt_template = self.episode_generation_prompt
                content_key = "conversation"
                time_key = "conversation_start_time"
            default_title = "Conversation Episode"
        else:
            pass

        # Extract earliest timestamp for context
        start_time = self._parse_timestamp(first_memcell.timestamp)
        start_time_str = self._format_timestamp(start_time)

        # Combine all content texts
        combined_content = "\n\n".join(all_content_text)

        # 构建 prompt
        if use_group_prompt:
            for i in range(5):
                try:
                    format_params = {
                        time_key: start_time_str,
                        content_key: combined_content,
                        "custom_instructions": self.default_custom_instructions,
                    }
                    prompt = prompt_template.format(**format_params)
                    response = await self.llm_provider.generate(prompt)
                    # 首先尝试提取代码块中的JSON
                    if '```json' in response:
                        # 提取代码块中的JSON内容
                        start = response.find('```json') + 7
                        end = response.find('```', start)
                        if end > start:
                            json_str = response[start:end].strip()
                            data = json.loads(json_str)
                        else:
                            # 尝试解析整个响应为JSON
                            data = json.loads(response)
                    else:
                        # 尝试匹配包含title和content的JSON对象
                        json_match = re.search(
                            r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}', response, re.DOTALL
                        )
                        if json_match:
                            data = json.loads(json_match.group())
                        else:
                            # 尝试解析整个响应为JSON
                            data = json.loads(response)
                    break
                except Exception as e:
                    print('retry: ', i)
                    if i == 4:
                        raise Exception("Episode memory extraction failed")
                    continue
            # Ensure we have required fields with fallback defaults
            if "title" not in data:
                data["title"] = default_title
            if "content" not in data:
                data["content"] = combined_content
            if "summary" not in data:
                # Generate a basic summary from content if not provided
                data["summary"] = data["content"]

            title = data["title"]
            content = data["content"]
            summary = data["summary"]

            # GROUP_EPISODE_GENERATION_PROMPT 模式：将情景记忆存储到 MemCell 中，返回 MemCell
            # 更新 MemCell 的 episode 字段
            for memcell in request.memcell_list:
                memcell.subject = title
                memcell.episode = content

            if use_semantic_extraction:
                await self._trigger_semantic_extraction_for_memcell_async(
                    first_memcell, request
                )

            # 返回第一个 MemCell（已经包含了情景记忆内容）
            return first_memcell
        else:
            format_params = {
                time_key: start_time_str,
                content_key: combined_content,
                "custom_instructions": self.default_custom_instructions,
            }

            participants = []
            [
                participants.extend(memcell.participants)
                for memcell in request.memcell_list
            ]
            if not participants:
                participants = request.participants
            if not participants:
                participants = []

            all_memories = []
            if participants:
                all_original_data = []
                [
                    all_original_data.extend(memcell.original_data)
                    for memcell in request.memcell_list
                ]
                participants_name_map = self.get_speaker_name_map(all_original_data)
                [
                    participants_name_map.update(
                        self._extract_participant_name_map(memcell.original_data)
                    )
                    for memcell in request.memcell_list
                ]

                # 并发生成每个参与者的episode memory
                async def generate_memory_for_user(
                    user_id: str, user_name: str
                ) -> EpisodeMemory:
                    user_format_params = format_params.copy()
                    user_format_params["user_name"] = user_name
                    prompt = prompt_template.format(**user_format_params)
                    response = await self.llm_provider.generate(prompt)

                    # 首先尝试提取代码块中的JSON
                    if '```json' in response:
                        # 提取代码块中的JSON内容
                        start = response.find('```json') + 7
                        end = response.find('```', start)
                        if end > start:
                            json_str = response[start:end].strip()
                            data = json.loads(json_str)
                        else:
                            # 尝试解析整个响应为JSON
                            data = json.loads(response)
                    else:
                        # 尝试匹配包含title和content的JSON对象
                        json_match = re.search(
                            r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}',
                            response,
                            re.DOTALL,
                        )
                        if json_match:
                            data = json.loads(json_match.group())
                        else:
                            # 尝试解析整个响应为JSON
                            data = json.loads(response)

                    # Ensure we have required fields with fallback defaults
                    if "title" not in data:
                        data["title"] = default_title
                    if "content" not in data:
                        data["content"] = combined_content
                    if "summary" not in data:
                        # Generate a basic summary from content if not provided
                        data["summary"] = "\n".join(
                            [memcell.summary for memcell in request.memcell_list]
                        )

                    title = data["title"]
                    content = data["content"]
                    summary = data["summary"]

                    return EpisodeMemory(
                        memory_type=MemoryType.EPISODE_SUMMARY,
                        user_id=user_id,
                        ori_event_id_list=[
                            memcell.event_id for memcell in request.memcell_list
                        ],
                        timestamp=start_time,
                        subject=title,
                        summary=summary,
                        episode=content,
                        group_id=request.group_id,
                        participants=participants,
                        type=getattr(first_memcell, 'type', None),
                        memcell_event_id_list=[
                            memcell.event_id for memcell in request.memcell_list
                        ],
                    )

                # 并发执行所有参与者的memory生成
                participant_memories = await asyncio.gather(
                    *[
                        generate_memory_for_user(
                            user_id, participants_name_map.get(user_id, user_id)
                        )
                        for user_id in participants
                    ],
                    return_exceptions=True,
                )

                # 处理结果，过滤掉异常
                for memory in participant_memories:
                    if isinstance(memory, EpisodeMemory):
                        all_memories.append(memory)
                    else:
                        print(
                            f"[EpisodicMemoryExtractor] Error generating memory: {memory}"
                        )

            for user_id in request.user_id_list:
                if user_id not in participants:
                    memory = EpisodeMemory(
                        memory_type=MemoryType.EPISODE_SUMMARY,
                        user_id=user_id,
                        ori_event_id_list=[
                            memcell.event_id for memcell in request.memcell_list
                        ],
                        timestamp=start_time,
                        subject=title,
                        summary="\n".join(
                            [memcell.summary for memcell in request.memcell_list]
                        ),
                        episode="\n".join(
                            [memcell.episode for memcell in request.memcell_list]
                        ),
                        group_id=request.group_id,
                        participants=participants,
                        type=getattr(first_memcell, 'type', None),
                        memcell_event_id_list=[
                            memcell.event_id for memcell in request.memcell_list
                        ],
                    )
                    all_memories.append(memory)
            # 异步触发语义记忆提取，不影响主流程
            # if all_memories:
            #     asyncio.create_task(self._trigger_semantic_extraction_async(all_memories, request))
            if use_semantic_extraction:
                await self._trigger_semantic_extraction_async(all_memories, request)
            return all_memories
