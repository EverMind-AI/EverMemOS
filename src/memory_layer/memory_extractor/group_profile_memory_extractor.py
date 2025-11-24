"""Group Profile Memory Extraction for EverMemOS."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import hashlib
import os

from ..llm.llm_provider import LLMProvider
from .base_memory_extractor import MemoryExtractor, MemoryExtractRequest
from ..types import Memory, MemoryType, MemCell
from common_utils.datetime_utils import (
    get_now_with_timezone,
    from_timestamp,
    from_iso_format,
    timezone,
)
from core.observation.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# 工具函数
# ============================================================================


def convert_to_datetime(timestamp, fallback_timestamp=None) -> datetime:
    """
    Convert various timestamp formats to datetime object with consistent timezone.

    Args:
        timestamp: The timestamp to convert
        fallback_timestamp: Fallback timestamp to use instead of current time

    Returns:
        datetime object with project consistent timezone
    """
    if isinstance(timestamp, datetime):
        # 确保时区一致性
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone)
        return timestamp.astimezone(timezone)
    elif isinstance(timestamp, (int, float)):
        # 使用公用函数，自动处理时区和精度识别
        return from_timestamp(timestamp)
    elif isinstance(timestamp, str):
        try:
            # 使用公用函数，统一时区处理
            return from_iso_format(timestamp)
        except Exception as e:
            logger.exception(
                f"Failed to parse timestamp: {timestamp}, error: {e}, using fallback"
            )
            return fallback_timestamp if fallback_timestamp else get_now_with_timezone()
    else:
        logger.exception(f"Unknown timestamp format: {timestamp}, using fallback")
        return fallback_timestamp if fallback_timestamp else get_now_with_timezone()


# ============================================================================
# 数据模型 - 保留在主文件，确保引用路径不变
# ============================================================================


class GroupRole(Enum):
    """7 key group roles in English."""

    DECISION_MAKER = "decision_maker"
    OPINION_LEADER = "opinion_leader"
    TOPIC_INITIATOR = "topic_initiator"
    EXECUTION_PROMOTER = "execution_promoter"
    CORE_CONTRIBUTOR = "core_contributor"
    COORDINATOR = "coordinator"
    INFO_SUMMARIZER = "info_summarizer"


class TopicStatus(Enum):
    """Topic status options."""

    EXPLORING = "exploring"
    DISAGREEMENT = "disagreement"
    CONSENSUS = "consensus"
    IMPLEMENTED = "implemented"


@dataclass
class TopicInfo:
    """
    Topic information for storage and output.

    包含 topic 的所有信息，包括 evidences 和 confidence。
    """

    name: str  # 话题名 (短语化标签)
    summary: str  # 一句话概述
    status: str  # exploring/disagreement/consensus/implemented
    last_active_at: datetime  # 最近活跃时间 (=updateTime)
    id: Optional[str] = None  # 话题唯一ID (系统生成，LLM不需要提供)
    update_type: Optional[str] = None  # "new" | "update" (仅用于增量更新时)
    old_topic_id: Optional[str] = None  # 更新时指向老topic (仅用于增量更新时)
    evidences: Optional[List[str]] = field(default_factory=list)  # memcell_ids 作为证据
    confidence: Optional[str] = None  # "strong" | "weak" - 置信度

    @classmethod
    def create_with_id(
        cls,
        name: str,
        summary: str,
        status: str,
        last_active_at: datetime,
        id: Optional[str] = None,
    ):
        """Create TopicInfo with generated or provided ID."""
        if not id:
            topic_id = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
            id = f"topic_{topic_id}"
        return cls(
            id=id,
            name=name,
            summary=summary,
            status=status,
            last_active_at=last_active_at,
        )


@dataclass
class GroupProfileMemory(Memory):
    """
    Group Profile Memory aligned with design document.

    Contains group core information extracted from conversations.
    Evidences are now stored within topics and roles instead of separately.
    """

    # 新增字段，不与基类冲突
    group_name: Optional[str] = None

    # 提取结果（包含 strong + weak，按 last_active_at 排序，限制为 max_topics 个）
    # topics 包含 evidences 和 confidence
    topics: Optional[List[TopicInfo]] = field(default_factory=list)
    # roles 的每个 assignment 包含 evidences 和 confidence
    # 格式: role -> [{"user_id": "xxx", "user_name": "xxx", "confidence": "strong|weak", "evidences": [...]}]
    roles: Optional[Dict[str, List[Dict[str, str]]]] = field(default_factory=dict)

    # 注意：summary 和 group_id 已经在基类中定义为 Optional，这里不需要重复定义

    def __post_init__(self):
        """Set memory_type to GROUP_PROFILE and call parent __post_init__."""
        self.memory_type = MemoryType.GROUP_PROFILE
        # 确保 topics 和 roles 不为 None，防止历史数据或异常情况导致的 None 值
        if self.topics is None:
            self.topics = []
        if self.roles is None:
            self.roles = {}
        super().__post_init__()


@dataclass
class GroupProfileMemoryExtractRequest(MemoryExtractRequest):
    """
    Request for group profile memory extraction.
    
    Group Profile 提取也可能需要处理多个 MemCell (来自聚类),
    因此也提供 memcell_list 支持
    """
    # 覆盖基类字段,可选的单个 memcell
    memcell: Optional[MemCell] = None
    
    # Group Profile 特有字段
    memcell_list: Optional[List[MemCell]] = None
    user_id_list: Optional[List[str]] = None
    
    def __post_init__(self):
        # 如果提供了 memcell_list,则使用它;否则使用单个 memcell
        if self.memcell_list is None and self.memcell is not None:
            self.memcell_list = [self.memcell]
        elif self.memcell_list is None:
            self.memcell_list = []


# ============================================================================
# 主提取器类 - 保留核心逻辑
# ============================================================================


class GroupProfileMemoryExtractor(MemoryExtractor):
    """
    Extractor for group profile information from conversations.

    Uses helper processors for data processing, topic/role management, and LLM interaction.
    Core business logic remains in this class.
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        conversation_source: str = "original",
        max_topics: int = 10,
    ):
        """
        初始化群组画像提取器

        Args:
            llm_provider: LLM提供者实例
            conversation_source: 对话来源，"original" 或 "episode"
            max_topics: 最大话题数量
        """
        super().__init__(MemoryType.GROUP_PROFILE)
        self.llm_provider = llm_provider
        self.conversation_source = conversation_source
        self.max_topics = max_topics

        # 延迟初始化辅助处理器
        self._data_processor = None
        self._topic_processor = None
        self._role_processor = None
        self._llm_handler = None

    # ========== 懒加载辅助处理器 ==========

    @property
    def data_processor(self):
        """Lazy load data processor."""
        if self._data_processor is None:
            from .group_profile.data_processor import GroupProfileDataProcessor

            self._data_processor = GroupProfileDataProcessor(self.conversation_source)
        return self._data_processor

    @property
    def topic_processor(self):
        """Lazy load topic processor."""
        if self._topic_processor is None:
            from .group_profile.topic_processor import TopicProcessor

            self._topic_processor = TopicProcessor(self.data_processor)
        return self._topic_processor

    @property
    def role_processor(self):
        """Lazy load role processor."""
        if self._role_processor is None:
            from .group_profile.role_processor import RoleProcessor

            self._role_processor = RoleProcessor(self.data_processor)
        return self._role_processor

    @property
    def llm_handler(self):
        """Lazy load LLM handler."""
        if self._llm_handler is None:
            from .group_profile.llm_handler import GroupProfileLLMHandler

            self._llm_handler = GroupProfileLLMHandler(
                self.llm_provider, self.max_topics
            )
        return self._llm_handler

    # ========== 业务逻辑方法 ==========

    def _filter_group(
        self, group_name: Optional[str], user_id_list: Optional[List[str]]
    ) -> bool:
        """
        Filter groups that should not be processed.

        Args:
            group_name: Group name to check
            user_id_list: List of user IDs (currently unused but kept for API compatibility)

        Returns:
            True if group should be filtered out (not processed), False otherwise
        """
        # 根据环境变量ENV控制过滤逻辑
        env_value = os.getenv('IGNORE_GROUP_NAME_FILTER', '').lower()

        if env_value == 'true':
            # 如果ENV变量为true，则不过滤任何群组
            return False
        else:
            # 否则执行原来的过滤逻辑
            if group_name:
                return False
            return True

    # ========== 核心提取方法 ==========

    async def extract_memory(
        self, request: GroupProfileMemoryExtractRequest
    ) -> Optional[List[GroupProfileMemory]]:
        """
        Extract group profile memory from conversation memcells.

        【核心业务流程】通过组合各个processor完成提取任务

        Args:
            request: Extract request containing memcells and related info

        Returns:
            List containing a single GroupProfileMemory object, or None
        """
        # ===== 1. 前置检查 =====
        if not request.memcell_list:
            return None

        group_id = request.group_id or ""
        group_name = request.group_name or ""
        memcell_list = request.memcell_list

        # 业务过滤逻辑
        if self._filter_group(group_name, request.user_id_list):
            logger.info(
                f"[GroupProfileMemoryExtractor] Skipping group '{group_name}' - filtered out"
            )
            return None

        # ===== 2. 提取历史画像和构建对话文本 =====
        existing_profile = self.data_processor.extract_existing_group_profile(
            request.old_memory_list
        )
        conversation_text = self.data_processor.combine_conversation_text_with_ids(
            memcell_list
        )

        # ===== 3. 计算时间跨度 =====
        start_time = convert_to_datetime(min(mc.timestamp for mc in memcell_list))
        end_time = convert_to_datetime(max(mc.timestamp for mc in memcell_list))
        timespan = f"{start_time.date()} to {end_time.date()}"

        try:
            # ===== 4. 执行LLM并行分析 =====
            logger.info(
                f"[GroupProfileMemoryExtractor] Executing parallel analysis for group: {group_name}"
            )

            parsed_data = await self.llm_handler.execute_parallel_analysis(
                conversation_text=conversation_text,
                group_id=group_id,
                group_name=group_name,
                memcell_list=memcell_list,
                existing_profile=existing_profile,
                user_organization=None,
                timespan=timespan,
            )

            if not parsed_data:
                return None

            # ===== 5. 收集有效memcell IDs =====
            valid_memcell_ids = set(
                str(mc.event_id)
                for mc in memcell_list
                if hasattr(mc, 'event_id') and mc.event_id
            )
            logger.debug(
                f"[extract_memory] Valid memcell IDs count: {len(valid_memcell_ids)}"
            )

            # ===== 6. 处理话题 =====
            raw_topics = parsed_data.get("topics", [])
            existing_topics = (
                existing_profile.get("topics", []) if existing_profile else []
            )

            all_topics = self.topic_processor.apply_topic_incremental_updates(
                llm_topics=raw_topics,
                existing_topics_with_evidences=existing_topics,
                memcell_list=memcell_list,
                valid_memcell_ids=valid_memcell_ids,
                max_topics=self.max_topics,
            )
            logger.info(
                f"[extract_memory] Processed {len(all_topics)} topics (strong + weak)"
            )

            # ===== 7. 处理角色 =====
            raw_roles = parsed_data.get("roles", {})
            existing_roles = (
                existing_profile.get("roles", {}) if existing_profile else {}
            )

            # 构建综合speaker映射
            comprehensive_mapping = (
                self.data_processor.get_comprehensive_speaker_mapping(
                    memcell_list, existing_roles
                )
            )

            all_roles = self.role_processor.process_roles_with_evidences(
                role_data=raw_roles,
                speaker_mapping=comprehensive_mapping,
                existing_roles=existing_roles,
                valid_memcell_ids=valid_memcell_ids,
                memcell_list=memcell_list,
            )
            logger.info(
                f"[extract_memory] Processed roles with {sum(len(v) for v in all_roles.values())} total assignments"
            )

            # ===== 8. 组装最终结果 =====
            group_profile = GroupProfileMemory(
                memory_type=MemoryType.GROUP_PROFILE,
                user_id="",
                timestamp=get_now_with_timezone(),
                ori_event_id_list=[
                    str(mc.event_id) for mc in memcell_list if hasattr(mc, 'event_id')
                ],
                group_id=group_id,
                group_name=group_name,
                topics=all_topics,  # All topics (strong + weak) with evidences, sorted by last_active_at
                roles=all_roles,  # All roles (strong + weak, strong first) with evidences
                summary=parsed_data.get("summary", ""),
                subject=parsed_data.get("subject", "not_found"),
            )

            return [group_profile]

        except Exception as e:
            logger.error(
                f"[GroupProfileMemoryExtractor] Extraction error: {e}", exc_info=True
            )
            return None


# ============================================================================
# 对外API声明
# ============================================================================

__all__ = [
    # 主类
    'GroupProfileMemoryExtractor',
    'GroupProfileMemoryExtractRequest',
    # 数据模型
    'GroupProfileMemory',
    'TopicInfo',
    # 枚举
    'GroupRole',
    'TopicStatus',
    # 工具函数
    'convert_to_datetime',
]
