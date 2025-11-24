"""
EpisodicMemory ES转换器

负责将MongoDB的EpisodicMemory文档转换为Elasticsearch的EpisodicMemoryDoc文档。
"""

from typing import List
import jieba
from core.oxm.es.base_converter import BaseEsConverter
from core.observation.logger import get_logger

# EpisodicMemory类型不再需要导入，因为参数类型已经简化为Any
from core.nlp.stopwords_utils import filter_stopwords
from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
    EpisodicMemoryDoc,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory as MongoEpisodicMemory,
)

logger = get_logger(__name__)


class EpisodicMemoryConverter(BaseEsConverter[EpisodicMemoryDoc]):
    """
    EpisodicMemory转换器

    将MongoDB的EpisodicMemory文档转换为Elasticsearch的EpisodicMemoryDoc文档。
    主要处理字段映射、搜索内容构建和数据格式转换。
    ES文档类型自动从泛型 BaseEsConverter[EpisodicMemoryDoc] 中获取。
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoEpisodicMemory) -> EpisodicMemoryDoc:
        """
        从MongoDB EpisodicMemory文档转换为ES EpisodicMemoryDoc实例

        使用场景：
        - ES索引重建时，从MongoDB文档转换为ES文档
        - 数据同步时，确保MongoDB数据正确映射到ES字段
        - 处理字段映射和数据格式转换

        Args:
            source_doc: MongoDB的EpisodicMemory文档实例

        Returns:
            EpisodicMemoryDoc: ES文档实例，可直接用于索引

        Raises:
            Exception: 当转换过程中发生错误时抛出异常
        """
        # 基本验证
        if source_doc is None:
            raise ValueError("MongoDB文档不能为空")

        try:
            # 构建搜索内容列表，用于BM25检索
            search_content = cls._build_search_content(source_doc)

            # 创建ES文档实例
            es_doc = EpisodicMemoryDoc(
                # 基础标识字段
                event_id=(
                    str(source_doc.id)
                    if hasattr(source_doc, 'id') and source_doc.id
                    else ""
                ),
                user_id=source_doc.user_id,
                user_name=getattr(source_doc, 'user_name', None),
                # 时间字段
                timestamp=source_doc.timestamp,
                # 核心内容字段
                title=getattr(
                    source_doc, 'subject', None
                ),  # MongoDB的subject映射到ES的title
                episode=source_doc.episode,
                search_content=search_content,  # BM25搜索的核心字段
                summary=getattr(source_doc, 'summary', None),
                # 分类和标签字段
                group_id=getattr(source_doc, 'group_id', None),
                participants=getattr(source_doc, 'participants', None),
                type=getattr(source_doc, 'type', None),
                keywords=getattr(source_doc, 'keywords', None),
                linked_entities=getattr(source_doc, 'linked_entities', None),
                # MongoDB特有字段
                subject=getattr(source_doc, 'subject', None),
                memcell_event_id_list=getattr(
                    source_doc, 'memcell_event_id_list', None
                ),
                # 扩展字段
                extend=getattr(source_doc, 'extend', None),
                # 审计字段
                created_at=getattr(source_doc, 'created_at', None),
                updated_at=getattr(source_doc, 'updated_at', None),
            )

            return es_doc

        except Exception as e:
            logger.error("从MongoDB文档转换为ES文档失败: %s", e)
            raise

    @classmethod
    def _build_search_content(cls, source_doc: MongoEpisodicMemory) -> List[str]:
        """
        构建搜索内容列表

        将MongoDB文档中的多个文本字段组合并使用jieba分词处理，
        生成用于BM25检索的搜索内容列表。

        Args:
            source_doc: MongoDB的EpisodicMemory文档实例

        Returns:
            List[str]: jieba分词后的搜索内容列表
        """
        text_content = []

        # 收集所有文本内容 - 包含 subject, summary, episode
        if hasattr(source_doc, 'subject') and source_doc.subject:
            text_content.append(source_doc.subject)
        
        if hasattr(source_doc, 'summary') and source_doc.summary:
            text_content.append(source_doc.summary)
        
        if hasattr(source_doc, 'episode') and source_doc.episode:
            text_content.append(source_doc.episode)

        # 将所有文本内容合并并使用jieba分词
        combined_text = ' '.join(text_content)
        search_content = list(jieba.cut(combined_text))

        # 过滤空字符串
        query_words = filter_stopwords(search_content, min_length=2)

        search_content = [word.strip() for word in query_words if word.strip()]

        return search_content

    @classmethod
    def from_memory(cls, episode_memory) -> EpisodicMemoryDoc:
        """
        从Memory对象转换为ES EpisodicMemoryDoc实例

        !!!!!!!!!!!!后面干掉！！！！！！！！！！！！！
        后面就这个原则，es完全从mongodb里面派生，就一个single source of truth 。


        专门用于处理从memory_manager获取的Memory对象，
        包含jieba分词处理和字段映射逻辑。

        Args:
            episode_memory: Memory对象实例

        Returns:
            EpisodicMemoryDoc: ES文档实例，可直接用于索引

        Raises:
            Exception: 当转换过程中发生错误时抛出异常
        """
        raise NotImplementedError("from_memory方法不再使用，请使用 from_mongo 方法从MongoDB文档转换")
