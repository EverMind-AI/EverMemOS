"""
Lite Model Field Extractor - 运行时动态提取索引字段

通过 Python 反射自动提取 Document 类的所有索引字段和查询字段，
无需手动维护 Lite 类代码。当第三方修改索引后，自动适配。
"""

from typing import Type, Set, Any, Dict
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from beanie import Indexed
import inspect

from core.observation.logger import get_logger

logger = get_logger(__name__)


class LiteModelExtractor:
    """
    Lite Model Field Extractor

    运行时动态提取 Document 的索引字段和查询字段，构建 Lite 版本数据。

    提取规则：
    1. 所有 Indexed 标记的字段
    2. Settings.indexes 中定义的索引字段
    3. Settings.query_fields 中配置的查询字段（无索引但用于查询）
    4. 审计字段：id, created_at, updated_at
    5. 软删除字段：deleted_at, deleted_by, deleted_id（如果存在）

    注意：query_fields 用于那些没有建索引但在查询中使用的字段
    """

    # 始终包含的系统字段
    SYSTEM_FIELDS = {"id", "created_at", "updated_at", "revision_id"}

    # 软删除字段（如果 Document 支持软删除）
    SOFT_DELETE_FIELDS = {"deleted_at", "deleted_by", "deleted_id"}

    @classmethod
    def extract_indexed_fields(cls, document_class: Type[BaseModel]) -> Set[str]:
        """
        提取 Document 类的所有索引字段和查询字段

        Args:
            document_class: Beanie Document 类

        Returns:
            Set[str]: 索引字段 + 查询字段名称集合
        """
        indexed_fields = set()

        # 1. 始终包含系统字段
        indexed_fields.update(cls.SYSTEM_FIELDS)

        # 2. 检查是否支持软删除（有 deleted_at 字段）
        if hasattr(document_class, "deleted_at"):
            indexed_fields.update(cls.SOFT_DELETE_FIELDS)

        # 3. 从字段注解中提取 Indexed 字段
        for field_name, field_info in document_class.model_fields.items():
            # 检查是否是 Indexed 类型
            if cls._is_indexed_field(field_info):
                indexed_fields.add(field_name)

        # 4. 从 Settings.indexes 中提取索引字段
        if hasattr(document_class, "Settings") and hasattr(document_class.Settings, "indexes"):
            for index_model in document_class.Settings.indexes:
                # IndexModel 的 document 属性返回完整索引规范（SON 对象）
                # 需要从 'key' 字段中提取实际的字段名
                if hasattr(index_model, "document"):
                    index_spec = index_model.document
                    # index_spec["key"] 是一个 SON 对象，包含 (field_name, direction) 对
                    if "key" in index_spec:
                        for field_name in index_spec["key"].keys():
                            indexed_fields.add(field_name)

        # 5. 从 Settings.query_fields 中提取查询字段（无索引但用于查询）
        if hasattr(document_class, "Settings") and hasattr(document_class.Settings, "query_fields"):
            query_fields = document_class.Settings.query_fields
            if query_fields:
                indexed_fields.update(query_fields)
                logger.debug(f"📋 Added {len(query_fields)} query fields (no index): {sorted(query_fields)}")

        logger.debug(f"📋 Extracted {len(indexed_fields)} total fields for {document_class.__name__}: {sorted(indexed_fields)}")
        return indexed_fields

    @classmethod
    def _is_indexed_field(cls, field_info: FieldInfo) -> bool:
        """
        检查字段是否是 Indexed 类型

        Args:
            field_info: Pydantic FieldInfo

        Returns:
            bool: 是否是索引字段
        """
        # 检查 annotation 是否包含 Indexed
        annotation = field_info.annotation

        # 处理 Optional[Indexed[...]] 的情况
        if hasattr(annotation, "__origin__"):
            # 获取泛型参数
            args = getattr(annotation, "__args__", ())
            for arg in args:
                if cls._is_indexed_type(arg):
                    return True

        # 直接检查是否是 Indexed 类型
        return cls._is_indexed_type(annotation)

    @classmethod
    def _is_indexed_type(cls, type_annotation: Any) -> bool:
        """
        检查类型是否是 Indexed

        Args:
            type_annotation: 类型注解

        Returns:
            bool: 是否是 Indexed 类型
        """
        # 检查是否是 Indexed 泛型
        if hasattr(type_annotation, "__origin__"):
            origin = type_annotation.__origin__
            # Indexed 在 beanie 中的实现
            if origin is not None and "Indexed" in str(origin):
                return True

        # 检查类型名称
        type_str = str(type_annotation)
        return "Indexed" in type_str

    @classmethod
    def extract_lite_data(cls, document: BaseModel, indexed_fields: Set[str]) -> Dict[str, Any]:
        """
        从完整 Document 提取 Lite 版本数据（只包含索引字段）

        Args:
            document: 完整的 Document 实例
            indexed_fields: 索引字段集合

        Returns:
            Dict[str, Any]: 只包含索引字段的字典
        """
        # Exclude Beanie internal fields that might be ExpressionField objects
        # These fields should not be serialized before the document is inserted
        exclude_fields = {'_id', 'id', 'revision_id'}

        try:
            full_data = document.model_dump(mode="python", exclude=exclude_fields)
        except Exception as e:
            # If model_dump fails, try to extract fields manually
            logger.warning(f"⚠️  model_dump failed, falling back to manual extraction: {e}")
            full_data = {}
            for field_name in document.model_fields.keys():
                if field_name not in exclude_fields:
                    try:
                        value = getattr(document, field_name, None)
                        # Skip ExpressionField objects
                        if value is not None and 'ExpressionField' not in str(type(value)):
                            full_data[field_name] = value
                    except Exception:
                        pass

        lite_data = {}

        for field_name in indexed_fields:
            if field_name in full_data:
                lite_data[field_name] = full_data[field_name]

        logger.debug(f"📦 Extracted lite data with {len(lite_data)} fields (from {len(full_data)} total fields)")
        return lite_data

    @classmethod
    def create_lite_document(cls, document: BaseModel, indexed_fields: Set[str]) -> BaseModel:
        """
        创建 Lite 版本的 Document 实例（只包含索引字段）

        Args:
            document: 完整的 Document 实例
            indexed_fields: 索引字段集合

        Returns:
            BaseModel: Lite 版本的 Document 实例
        """
        lite_data = cls.extract_lite_data(document, indexed_fields)

        # 使用相同的 Document 类创建实例，但只包含索引字段
        # Pydantic 会自动处理缺失的可选字段
        lite_document = document.__class__.model_validate(lite_data)

        return lite_document


__all__ = ["LiteModelExtractor"]
