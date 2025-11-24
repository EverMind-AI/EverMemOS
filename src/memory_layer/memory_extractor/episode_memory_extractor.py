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
from ..types import MemoryType, Memory, RawDataType, MemCell

from common_utils.datetime_utils import get_now_with_timezone
from agentic_layer.vectorize_service import get_vectorize_service

from core.observation.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EpisodeMemoryExtractRequest(MemoryExtractRequest):
    """Episode 提取请求（继承自基类）"""
    pass


class EpisodeMemoryExtractor(MemoryExtractor):
    """
    Episode 记忆提取器 - 只负责从 MemCell 中提取 Episode
    
    职责：
    1. 从 MemCell 的 original_data 中提取群组 Episode
    2. 从 MemCell 的 original_data 中提取个人 Episode
    
    不包含：
    - Semantic 提取（由 SemanticMemoryExtractor 负责）
    - EventLog 提取（由 EventLogExtractor 负责）
    """
    def __init__(
        self, llm_provider: LLMProvider | None = None, use_eval_prompts: bool = False
    ):
        super().__init__(MemoryType.EPISODIC_MEMORY)
        self.llm_provider = llm_provider
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

    
    
    

    async def _extract_episode(
        self,
        request: EpisodeMemoryExtractRequest,
        use_group_prompt: bool = False,
    ) -> Optional[Memory]:
        """
        提取 Episode 记忆（内部方法，单次提取）
        
        Args:
            request: Episode 提取请求（包含单个 memcell 和可选的 user_id）
            use_group_prompt: 是否使用群组提示词
                - True: 提取群组 Episode（user_id=None）
                - False: 提取个人 Episode（user_id 从 request.user_id 获取）
        
        Returns:
            Memory（包含 episode 字段）
        """
        logger.debug(f"📚 开始提取 Episode，use_group_prompt={use_group_prompt}")

        memcell = request.memcell
        if not memcell:
            return None

        # 准备对话文本
        if memcell.type == RawDataType.CONVERSATION:
            conversation_text = self.get_conversation_json_text(memcell.original_data)

            # 选择提示词和参数
            if use_group_prompt:
                prompt_template = self.group_episode_generation_prompt
                content_key = "conversation"
                time_key = "conversation_start_time"
            else:
                prompt_template = self.episode_generation_prompt
                content_key = "conversation"
                time_key = "conversation_start_time"
            default_title = "Conversation Episode"
        else:
            return None

        # 时间戳格式化
        start_time = self._parse_timestamp(memcell.timestamp)
        start_time_str = self._format_timestamp(start_time)

        # 构建 prompt 参数
        format_params = {
            time_key: start_time_str,
            content_key: conversation_text,
            "custom_instructions": self.default_custom_instructions,
        }
        
        # 获取参与者信息
        participants_name_map = self.get_speaker_name_map(memcell.original_data)
        participants_name_map.update(
            self._extract_participant_name_map(memcell.original_data)
        )
        
        # 确定 user_id 和 user_name
        user_id = None
        user_name = None
        if use_group_prompt:
            # 群组模式：user_id 为 None，user_name 为 None
            user_id = None
            user_name = None
        else:
            # 个人模式：从 request.user_id 获取
            if request.user_id:
                user_id = request.user_id
                user_name = participants_name_map.get(user_id, user_id)
                format_params["user_name"] = user_name

        # 调用 LLM（带重试）
        data = None
        for i in range(5):
            try:
                prompt = prompt_template.format(**format_params)
                response = await self.llm_provider.generate(prompt)
                
                # 解析 JSON
                if '```json' in response:
                    start = response.find('```json') + 7
                    end = response.find('```', start)
                    if end > start:
                        json_str = response[start:end].strip()
                        data = json.loads(json_str)
                    else:
                        data = json.loads(response)
                else:
                    json_match = re.search(
                        r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}',
                        response,
                        re.DOTALL,
                    )
                    if json_match:
                        data = json.loads(json_match.group())
                    else:
                        data = json.loads(response)
                
                # 验证必需字段：title 和 content 必须存在
                if "title" not in data or not data["title"]:
                    raise ValueError("LLM 返回缺少 title 字段")
                if "content" not in data or not data["content"]:
                    raise ValueError("LLM 返回缺少 content 字段")
                
                # 验证通过，跳出重试循环
                break
            except Exception as e:
                logger.warning(f"Episode 提取重试 {i+1}/5: {e}")
                if i == 4:
                    raise Exception("Episode memory extraction failed after 5 retries")
                continue

        # summary 没有的话使用 content 前200字符作为默认值
        if "summary" not in data or not data["summary"]:
            data["summary"] = data["content"][:200]

        title = data["title"]
        content = data["content"]
        summary = data["summary"]
        
        # 收集参与者
        participants = memcell.participants if memcell.participants else []

        # 计算 Embedding
        embedding_data = await self._compute_embedding(content)

        # 创建 Memory 对象（统一返回类型）
        episode_memory = Memory(
            memory_type=MemoryType.EPISODIC_MEMORY,
            user_id=user_id,
            user_name=user_name,
            ori_event_id_list=[memcell.event_id],
            timestamp=start_time,
            subject=title,
            summary=summary,
            episode=content,
            group_id=request.group_id,
            participants=participants,
            type=memcell.type,
            memcell_event_id_list=[memcell.event_id],
            extend=embedding_data,  # 添加 embedding 到 extend 字段
        )

        logger.debug(f"✅ Episode 提取完成: subject='{title}'")
        return episode_memory
    
    async def extract_memory(self, request: MemoryExtractRequest) -> Optional[Memory]:
        """
        从 MemCell 提取 Episode 记忆（实现基类的抽象方法）
        
        自动根据 request.user_id 判断提取群组还是个人 Episode:
        - user_id=None: 提取群组 Episode（使用群组提示词）
        - user_id!=None: 提取个人 Episode（使用个人提示词，聚焦该用户视角）
        
        Args:
            request: 记忆提取请求，包含：
                - memcell: 要提取的 MemCell
                - user_id: 用户ID（None表示群组）
                - group_id: 群组ID
                - 其他可选字段
        
        Returns:
            Memory: Episode 记忆对象
                - 群组 Episode: user_id=None, episode 包含整个对话的全局视角
                - 个人 Episode: user_id=<user_id>, episode 包含该用户的个人视角
        """
        # 判断是群组还是个人 Episode
        is_group_episode = (request.user_id is None)
        
        logger.debug(
            f"[extract_memory] 提取 {'群组' if is_group_episode else '个人'} Episode, "
            f"user_id={request.user_id}, group_id={request.group_id}"
        )
        
        # 构建 EpisodeMemoryExtractRequest
        episode_request = EpisodeMemoryExtractRequest(
            memcell=request.memcell,
            user_id=request.user_id,
            group_id=request.group_id,
            group_name=request.group_name,
            participants=request.participants,
            old_memory_list=request.old_memory_list,
            user_organization=request.user_organization,
        )
        
        # 调用内部提取方法
        return await self._extract_episode(
            request=episode_request,
            use_group_prompt=is_group_episode,  # 群组用群组提示词，个人用个人提示词
        )
    
    async def _compute_embedding(self, text: str) -> Optional[dict]:
        """计算 Episode 文本的 Embedding"""
        try:
            if not text:
                return None
            
            vs = get_vectorize_service()
            vec = await vs.get_embedding(text)
            
            return {
                "embedding": vec.tolist() if hasattr(vec, "tolist") else list(vec),
                "vector_model": vs.get_model_name()  # 使用统一的 get_model_name() 方法
            }
        except Exception as e:
            logger.error(f"Episode Embedding 计算失败: {e}")
            return None
