from dataclasses import dataclass
from datetime import datetime
import time
import os
import asyncio
from typing import List, Optional

from core.observation.logger import get_logger

from .llm.llm_provider import LLMProvider
from .memcell_extractor.conv_memcell_extractor import ConvMemCellExtractor
from .memcell_extractor.base_memcell_extractor import RawData
from .memcell_extractor.conv_memcell_extractor import ConversationMemCellExtractRequest
from .types import MemCell, RawDataType, MemoryType, SemanticMemoryItem
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
from .memory_extractor.semantic_memory_extractor import SemanticMemoryExtractor
from .memcell_extractor.base_memcell_extractor import StatusResult


logger = get_logger(__name__)


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


@dataclass
class MemorizeOfflineRequest:
    memorize_from: datetime
    memorize_to: datetime


class MemoryManager:
    """
    记忆管理器 - 负责编排所有记忆提取流程
    
    职责：
    1. 提取 MemCell（边界检测 + 原始数据）
    2. 提取 Episode/Semantic/EventLog/Profile 等记忆（基于 MemCell 或 episode）
    3. 管理所有 Extractor 的生命周期
    4. 提供统一的记忆提取接口
    """
    def __init__(self):
        # 统一的 LLM Provider - 所有 extractor 共用
        self.llm_provider = LLMProvider(
            provider_type=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "Qwen3-235B"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY", "123"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
        )
        
        # Episode Extractor - 延迟初始化
        self._episode_extractor = None
# TODO:添加 username
    async def extract_memcell(
        self,
        history_raw_data_list: list[RawData],
        new_raw_data_list: list[RawData],
        raw_data_type: RawDataType,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        user_id_list: Optional[List[str]] = None,
        old_memory_list: Optional[List[Memory]] = None,
    ) -> tuple[Optional[MemCell], Optional[StatusResult]]:
        """
        提取 MemCell（边界检测 + 原始数据）
        
        Args:
            history_raw_data_list: 历史消息列表
            new_raw_data_list: 新消息列表
            raw_data_type: 数据类型
            group_id: 群组ID
            group_name: 群组名称
            user_id_list: 用户ID列表
            old_memory_list: 历史记忆列表
        
        Returns:
            (MemCell, StatusResult) 或 (None, StatusResult)
        """
        now = time.time()

        # 边界检测 + 创建 MemCell
        logger.debug(f"[MemoryManager] 开始边界检测并创建 MemCell")
        request = ConversationMemCellExtractRequest(
            history_raw_data_list,
            new_raw_data_list,
            user_id_list=user_id_list,
            group_id=group_id,
            group_name=group_name,
            old_memory_list=old_memory_list,
        )
        
        extractor = ConvMemCellExtractor(self.llm_provider)
        memcell, status_result = await extractor.extract_memcell(request)

        if not memcell:
            logger.debug(f"[MemoryManager] 边界检测：未到边界，等待更多消息")
            return None, status_result
        
        logger.info(
            f"[MemoryManager] ✅ MemCell 创建成功: "
            f"event_id={memcell.event_id}, "
            f"耗时: {time.time() - now:.2f}秒"
        )

        return memcell, status_result
    
# TODO:添加 username
    async def extract_memory(
        self,
        memcell: MemCell,
        memory_type: MemoryType,
        user_id: Optional[str] = None,  # None 表示群组记忆，有值表示个人记忆
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        old_memory_list: Optional[List[Memory]] = None,
        user_organization: Optional[List] = None,
        episode_memory: Optional[Memory] = None,  # 用于 Semantic 和 EventLog 提取
    ):
        """
        提取单个记忆
        
        Args:
            memcell: 单个 MemCell（记忆的原始数据容器）
            memory_type: 记忆类型
            user_id: 用户ID
                - None: 提取群组 Episode/群组 Profile
                - 有值: 提取个人 Episode/个人 Profile
            group_id: 群组ID
            group_name: 群组名称
            old_memory_list: 历史记忆列表
            user_organization: 用户组织信息
            episode_memory: Episode 记忆（用于提取 Semantic/EventLog）
        
        Returns:
            - EPISODIC_MEMORY: 返回 Memory（群组或个人）
            - SEMANTIC_MEMORY: 返回 List[SemanticMemoryItem]
            - PERSONAL_EVENT_LOG: 返回 EventLog
            - PROFILE/GROUP_PROFILE: 返回 Memory
        """
        # 根据 memory_type 枚举分发处理
        match memory_type:
            case MemoryType.EPISODIC_MEMORY:
                return await self._extract_episode(memcell, user_id, group_id)
            
            case MemoryType.SEMANTIC_MEMORY:
                return await self._extract_semantic(episode_memory)
            
            case MemoryType.PERSONAL_EVENT_LOG:
                return await self._extract_event_log(episode_memory)
            
            case MemoryType.PROFILE:
                return await self._extract_profile(
                    memcell, user_id, group_id, old_memory_list
                )
            
            case MemoryType.GROUP_PROFILE:
                return await self._extract_group_profile(
                    memcell, user_id, group_id, group_name, 
                    old_memory_list, user_organization
                )
            
            case _:
                logger.warning(f"[MemoryManager] 未知的 memory_type: {memory_type}")
                return None
    
    async def _extract_episode(
        self,
        memcell: MemCell,
        user_id: Optional[str],
        group_id: Optional[str],
    ) -> Optional[Memory]:
        """提取 Episode（群组或个人）"""
        if self._episode_extractor is None:
            self._episode_extractor = EpisodeMemoryExtractor(self.llm_provider)
        
        # 构建提取请求
        from .memory_extractor.base_memory_extractor import MemoryExtractRequest
        
        request = MemoryExtractRequest(
            memcell=memcell,
            user_id=user_id,  # None=群组，有值=个人
            group_id=group_id,
        )
        
        # 调用 extractor 的 extract_memory 方法
        # 它会根据 user_id 自动判断提取群组还是个人 Episode
        logger.debug(
            f"[MemoryManager] 提取 {'群组' if user_id is None else '个人'} Episode: user_id={user_id}"
        )
        
        return await self._episode_extractor.extract_memory(request)
    
    async def _extract_semantic(
        self,
        episode_memory: Optional[Memory],
    ) -> List[SemanticMemoryItem]:
        """提取 Semantic Memory"""
        if not episode_memory:
            logger.warning("[MemoryManager] 缺少 episode_memory，无法提取 Semantic")
            return []
        
        logger.debug(f"[MemoryManager] 为 Episode 提取 Semantic: user_id={episode_memory.user_id}")
        
        extractor = SemanticMemoryExtractor(llm_provider=self.llm_provider)
        return await extractor.generate_semantic_memories_for_episode(episode_memory)
    
    async def _extract_event_log(
        self,
        episode_memory: Optional[Memory],
    ):
        """提取 Event Log"""
        if not episode_memory:
            logger.warning("[MemoryManager] 缺少 episode_memory，无法提取 EventLog")
            return None
        
        logger.debug(f"[MemoryManager] 为 Episode 提取 EventLog: user_id={episode_memory.user_id}")
        
        extractor = EventLogExtractor(llm_provider=self.llm_provider)
        return await extractor.extract_event_log(
            episode_text=episode_memory.episode,
            timestamp=episode_memory.timestamp
        )
    
    async def _extract_profile(
        self,
        memcell: MemCell,
        user_id: Optional[str],
        group_id: Optional[str],
        old_memory_list: Optional[List[Memory]],
    ) -> Optional[Memory]:
        """提取 Profile"""
        if memcell.type != RawDataType.CONVERSATION:
            return None
        
        extractor = ProfileMemoryExtractor(self.llm_provider)
        request = ProfileMemoryExtractRequest(
            memcell_list=[memcell],
            user_id_list=[user_id] if user_id else [],
            group_id=group_id,
            old_memory_list=old_memory_list,
        )
        return await extractor.extract_memory(request)
    
    async def _extract_group_profile(
        self,
        memcell: MemCell,
        user_id: Optional[str],
        group_id: Optional[str],
        group_name: Optional[str],
        old_memory_list: Optional[List[Memory]],
        user_organization: Optional[List],
    ) -> Optional[Memory]:
        """提取 Group Profile"""
        extractor = GroupProfileMemoryExtractor(self.llm_provider)
        request = GroupProfileMemoryExtractRequest(
            memcell_list=[memcell],
            user_id_list=[user_id] if user_id else [],
            group_id=group_id,
            group_name=group_name,
            old_memory_list=old_memory_list,
            user_organization=user_organization,
        )
        return await extractor.extract_memory(request)
