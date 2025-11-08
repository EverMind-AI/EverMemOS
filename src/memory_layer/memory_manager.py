from dataclasses import dataclass
import datetime
import time

from core.observation.logger import get_logger

from .llm.llm_provider import LLMProvider
from .memcell_extractor.conv_memcell_extractor import ConvMemCellExtractor
from .memcell_extractor.base_memcell_extractor import RawData
from .memcell_extractor.conv_memcell_extractor import ConversationMemCellExtractRequest
from .types import MemCell
from .memory_extractor.episode_memory_extractor import (
    EpisodeMemoryExtractor,
    EpisodeMemoryExtractRequest,
    Memory,
)
from .memory_extractor.profile_memory_extractor import (
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
)
from .memory_extractor.group_profile_memory_extractor import (
    GroupProfileMemoryExtractor,
    GroupProfileMemoryExtractRequest,
)
from .memory_extractor.event_log_extractor import EventLogExtractor
import os
from .types import RawDataType, MemoryType
from .memcell_extractor.base_memcell_extractor import StatusResult
from typing import List, Optional


@dataclass
class MemorizeRequest:
    history_raw_data_list: list[RawData]
    new_raw_data_list: list[RawData]
    raw_data_type: RawDataType
    # 整个group全量的user_id列表
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    current_time: Optional[datetime] = None
    # 可选的提取控制参数
    enable_semantic_extraction: bool = True  # 是否提取语义记忆
    enable_event_log_extraction: bool = True  # 是否提取事件日志
    # 对话元数据字段（对应 ConversationMeta）
    scene: Optional[str] = None  # 场景标识符，如 "company"、"work" 等
    scene_desc: Optional[dict] = None  # 场景描述信息，如 {"bot_ids": ["aaa", "bbb"]}


@dataclass
class MemorizeOfflineRequest:
    memorize_from: datetime
    memorize_to: datetime


class MemoryManager:
    def __init__(self):
        # Conversation MemCell LLM Provider - 从环境变量读取配置
        self.conv_memcall_llm_provider = LLMProvider(
            provider_type=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv(
                "LLM_BASE_URL"
            ),
            api_key=os.getenv("LLM_API_KEY", "123"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        )
        
        # Event Log Extractor LLM Provider - 从环境变量读取配置
        self.event_log_llm_provider = LLMProvider(
            provider_type=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY", "123"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        )
        
        # Event Log Extractor - 延迟初始化
        self._event_log_extractor = None

        # Episode Memory Extractor LLM Provider - 从环境变量读取配置
        self.episode_memory_extractor_llm_provider = LLMProvider(
            provider_type=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv(
                "LLM_BASE_URL"
            ),
            api_key=os.getenv("LLM_API_KEY", "123"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        )

        # Profile Memory Extractor LLM Provider - 从环境变量读取配置
        self.profile_memory_extractor_llm_provider = LLMProvider(
            provider_type=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv(
                "LLM_BASE_URL"
            ),
            api_key=os.getenv("LLM_API_KEY", "123"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        )

        

    async def extract_memcell(
        self,
        history_raw_data_list: list[RawData],
        new_raw_data_list: list[RawData],
        raw_data_type: RawDataType,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        user_id_list: Optional[List[str]] = None,
        old_memory_list: Optional[List[Memory]] = None,
        enable_semantic_extraction: bool = True,
        enable_event_log_extraction: bool = True,
    ) -> tuple[Optional[MemCell], Optional[StatusResult]]:
        """
        提取 MemCell（包含可选的语义记忆和事件日志提取）
        
        Args:
            history_raw_data_list: 历史消息列表
            new_raw_data_list: 新消息列表
            raw_data_type: 数据类型
            group_id: 群组ID
            group_name: 群组名称
            user_id_list: 用户ID列表
            old_memory_list: 历史记忆列表
            enable_semantic_extraction: 是否提取语义记忆（默认True）
            enable_event_log_extraction: 是否提取事件日志（默认True）
            
        Returns:
            (MemCell, StatusResult) 或 (None, StatusResult)
        """
        logger = get_logger(__name__)
        now = time.time()
        
        # 1. 提取基础 MemCell（包括可选的语义记忆）
        request = ConversationMemCellExtractRequest(
            history_raw_data_list,
            new_raw_data_list,
            user_id_list=user_id_list,
            group_id=group_id,
            group_name=group_name,
            old_memory_list=old_memory_list,
        )
        extractor = ConvMemCellExtractor(self.conv_memcall_llm_provider)
        memcell, status_result = await extractor.extract_memcell(
            request, 
            use_semantic_extraction=enable_semantic_extraction
        )
        
        # 2. 如果成功提取 MemCell，且启用了 Event Log 提取
        if memcell and enable_event_log_extraction and hasattr(memcell, 'episode') and memcell.episode:
            if self._event_log_extractor is None:
                self._event_log_extractor = EventLogExtractor(llm_provider=self.event_log_llm_provider)
            
            logger.debug(f"开始提取 Event Log: {memcell.event_id}")
            event_log = await self._event_log_extractor.extract_event_log(
                episode_text=memcell.episode,
                timestamp=memcell.timestamp
            )
            
            if event_log:
                memcell.event_log = event_log
                logger.debug(f"Event Log 提取成功: {memcell.event_id}")
        
        logger.debug(
            f"提取MemCell完成, raw_data_type: {raw_data_type}, "
            f"semantic_extraction={enable_semantic_extraction}, "
            f"event_log_extraction={enable_event_log_extraction}, "
            f"耗时: {time.time() - now}秒"
        )
        
        return memcell, status_result

    async def extract_memory(
        self,
        memcell_list: list[MemCell],
        memory_type: MemoryType,
        user_ids: List[str],
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        old_memory_list: Optional[List[Memory]] = None,
        user_organization: Optional[List] = None,
    ) -> List[Memory]:
        extractor = None
        request = None

        if memory_type == MemoryType.EPISODE_SUMMARY:
            extractor = EpisodeMemoryExtractor(
                self.episode_memory_extractor_llm_provider
            )
            request = EpisodeMemoryExtractRequest(
                memcell_list=memcell_list,
                user_id_list=user_ids,
                group_id=group_id,
                old_memory_list=old_memory_list,
            )
        elif memory_type == MemoryType.PROFILE:
            if memcell_list[0].type == RawDataType.CONVERSATION:
                extractor = ProfileMemoryExtractor(
                    self.profile_memory_extractor_llm_provider
                )
                request = ProfileMemoryExtractRequest(
                    memcell_list=memcell_list,
                    user_id_list=user_ids,
                    group_id=group_id,
                    old_memory_list=old_memory_list,
                )
        elif memory_type == MemoryType.GROUP_PROFILE:
            extractor = GroupProfileMemoryExtractor(
                self.profile_memory_extractor_llm_provider
            )
            request = GroupProfileMemoryExtractRequest(
                memcell_list=memcell_list,
                user_id_list=user_ids,
                group_id=group_id,
                group_name=group_name,
                old_memory_list=old_memory_list,
                user_organization=None,
            )

        if extractor == None or request == None:
            return []
        return await extractor.extract_memory(request)
