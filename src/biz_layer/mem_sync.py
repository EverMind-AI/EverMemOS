"""语义记忆与事件日志同步服务

负责将统一的语义记忆与事件日志写入 Milvus / Elasticsearch。
"""

from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from infra_layer.adapters.out.persistence.document.memory.semantic_memory_record import (
    SemanticMemoryRecord,
)
from infra_layer.adapters.out.search.elasticsearch.converter.semantic_memory_converter import (
    SemanticMemoryConverter,
)
from infra_layer.adapters.out.search.milvus.converter.semantic_memory_milvus_converter import (
    SemanticMemoryMilvusConverter,
)
from infra_layer.adapters.out.persistence.document.memory.event_log_record import (
    EventLogRecord,
)
from infra_layer.adapters.out.search.elasticsearch.converter.event_log_converter import (
    EventLogConverter,
)
from infra_layer.adapters.out.search.milvus.converter.event_log_milvus_converter import (
    EventLogMilvusConverter,
)
from infra_layer.adapters.out.search.repository.semantic_memory_milvus_repository import (
    SemanticMemoryMilvusRepository,
)
from infra_layer.adapters.out.search.repository.event_log_milvus_repository import (
    EventLogMilvusRepository,
)
from infra_layer.adapters.out.search.repository.semantic_memory_es_repository import (
    SemanticMemoryEsRepository,
)
from infra_layer.adapters.out.search.repository.event_log_es_repository import (
    EventLogEsRepository,
)
from core.di import get_bean_by_type, service
from common_utils.datetime_utils import get_now_with_timezone

logger = logging.getLogger(__name__)


@service(name="memory_sync_service", primary=True)
class MemorySyncService:
    """语义记忆与事件日志同步服务"""

    def __init__(
        self,
        semantic_milvus_repo: Optional[SemanticMemoryMilvusRepository] = None,
        eventlog_milvus_repo: Optional[EventLogMilvusRepository] = None,
        semantic_es_repo: Optional[SemanticMemoryEsRepository] = None,
        eventlog_es_repo: Optional[EventLogEsRepository] = None,
    ):
        """初始化同步服务
        
        Args:
            semantic_milvus_repo: 语义记忆 Milvus 仓库实例（可选，不提供则从 DI 获取）
            eventlog_milvus_repo: 事件日志 Milvus 仓库实例（可选，不提供则从 DI 获取）
            semantic_es_repo: 语义记忆 ES 仓库实例（可选，不提供则从 DI 获取）
            eventlog_es_repo: 事件日志 ES 仓库实例（可选，不提供则从 DI 获取）
        """
        self.semantic_milvus_repo = semantic_milvus_repo or get_bean_by_type(
            SemanticMemoryMilvusRepository
        )
        self.eventlog_milvus_repo = eventlog_milvus_repo or get_bean_by_type(
            EventLogMilvusRepository
        )
        self.semantic_es_repo = semantic_es_repo or get_bean_by_type(
            SemanticMemoryEsRepository
        )
        self.eventlog_es_repo = eventlog_es_repo or get_bean_by_type(
            EventLogEsRepository
        )
        
        
        logger.info("MemorySyncService 初始化完成")

    @staticmethod
    def _normalize_datetime(value: Optional[datetime | str]) -> Optional[datetime]:
        """将 str/None 转为 datetime（仅日期字符串也支持）"""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d")
                except ValueError:
                    logger.warning("无法解析日期字符串: %s", value)
                    return None
        return None

    async def sync_semantic_memory(
        self, 
        semantic_memory: SemanticMemoryRecord,
        sync_to_es: bool = True,
        sync_to_milvus: bool = True
    ) -> Dict[str, int]:
        """同步单条语义记忆到 Milvus/ES
        
        Args:
            semantic_memory: SemanticMemoryRecord 文档对象
            sync_to_es: 是否同步到 ES（默认 True）
            sync_to_milvus: 是否同步到 Milvus（默认 True）
            
        Returns:
            同步统计信息 {"semantic_memory": 1}
        """
        stats = {"semantic_memory": 0, "es_records": 0}
        
        try:
            # 从 MongoDB 读取 embedding，如果没有则跳过
            if not semantic_memory.vector:
                logger.warning(f"语义记忆 {semantic_memory.id} 没有 embedding，跳过同步")
                return stats
            
            # 同步到 Milvus
            if sync_to_milvus:
                # 使用转换器生成 Milvus 实体
                milvus_entity = SemanticMemoryMilvusConverter.from_mongo(semantic_memory)
                await self.semantic_milvus_repo.insert(milvus_entity, flush=False)
                stats["semantic_memory"] += 1
                logger.debug(f"已同步语义记忆到 Milvus: {semantic_memory.id}")
            
            # 同步到 ES
            if sync_to_es:
                # 使用转换器生成正确的 ES 文档(包括 jieba 分词的 search_content)
                es_doc = SemanticMemoryConverter.from_mongo(semantic_memory)
                await es_doc.save()
                stats["es_records"] += 1
                logger.debug(f"已同步语义记忆到 ES: {semantic_memory.id}")
            
        except Exception as e:
            logger.error(f"同步语义记忆失败: {e}", exc_info=True)
            raise
        
        return stats

    async def sync_event_log(
        self,
        event_log: EventLogRecord,
        sync_to_es: bool = True,
        sync_to_milvus: bool = True
    ) -> Dict[str, int]:
        """同步单条事件日志到 Milvus/ES
        
        Args:
            event_log: EventLogRecord 文档对象
            sync_to_es: 是否同步到 ES（默认 True）
            sync_to_milvus: 是否同步到 Milvus（默认 True）
            
        Returns:
            同步统计信息 {"event_log": 1}
        """
        stats = {"event_log": 0, "es_records": 0}
        
        try:
            # 从 MongoDB 读取已有的 vector
            if not event_log.vector:
                logger.warning(f"事件日志 {event_log.id} 没有 embedding，跳过同步")
                return stats
            
            # 同步到 Milvus
            if sync_to_milvus:
                # 使用转换器生成 Milvus 实体
                milvus_entity = EventLogMilvusConverter.from_mongo(event_log)
                await self.eventlog_milvus_repo.insert(milvus_entity, flush=False)
                stats["event_log"] += 1
                logger.debug(f"已同步事件日志到 Milvus: {event_log.id}")
            
            # 同步到 ES
            if sync_to_es:
                # 使用转换器生成正确的 ES 文档(包括 jieba 分词的 search_content)
                es_doc = EventLogConverter.from_mongo(event_log)
                await es_doc.save()
                stats["es_records"] += 1
                logger.debug(f"已同步事件日志到 ES: {event_log.id}")
            
        except Exception as e:
            logger.error(f"同步事件日志失败: {e}", exc_info=True)
            raise
        
        return stats

    async def sync_batch_semantic_memories(
        self,
        semantic_memories: List[SemanticMemoryRecord],
        sync_to_es: bool = True,
        sync_to_milvus: bool = True
    ) -> Dict[str, int]:
        """批量同步语义记忆
        
        Args:
            semantic_memories: SemanticMemoryRecord 列表
            sync_to_es: 是否同步到 ES（默认 True）
            sync_to_milvus: 是否同步到 Milvus（默认 True）
            
        Returns:
            同步统计信息
        """
        total_stats = {"semantic_memory": 0, "es_records": 0}
        
        for sem_mem in semantic_memories:
            try:
                stats = await self.sync_semantic_memory(
                    sem_mem, 
                    sync_to_es=sync_to_es,
                    sync_to_milvus=sync_to_milvus
                )
                total_stats["semantic_memory"] += stats.get("semantic_memory", 0)
                total_stats["es_records"] += stats.get("es_records", 0)
            except Exception as e:
                logger.error(f"批量同步语义记忆失败: {sem_mem.id}, 错误: {e}", exc_info=True)
                # 不要静默吞掉异常
        
        
        logger.info(f"✅ 语义记忆 Milvus flush 完成: {total_stats['semantic_memory']} 条")
        
        return total_stats

    async def sync_batch_event_logs(
        self,
        event_logs: List[EventLogRecord],
        sync_to_es: bool = True,
        sync_to_milvus: bool = True
    ) -> Dict[str, int]:
        """批量同步事件日志
        
        Args:
            event_logs: EventLogRecord 列表
            sync_to_es: 是否同步到 ES（默认 True）
            sync_to_milvus: 是否同步到 Milvus（默认 True）
            
        Returns:
            同步统计信息
        """
        total_stats = {"event_log": 0, "es_records": 0}
        
        for evt_log in event_logs:
            try:
                stats = await self.sync_event_log(
                    evt_log,
                    sync_to_es=sync_to_es,
                    sync_to_milvus=sync_to_milvus
                )
                total_stats["event_log"] += stats.get("event_log", 0)
                total_stats["es_records"] += stats.get("es_records", 0)
            except Exception as e:
                logger.error(f"批量同步事件日志失败: {evt_log.id}, 错误: {e}", exc_info=True)
                # 不要静默吞掉异常，让它暴露出来
                raise
        
        
        logger.info(f"✅ 事件日志 Milvus flush 完成: {total_stats['event_log']} 条")
        
        return total_stats

