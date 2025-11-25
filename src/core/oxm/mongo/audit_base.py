"""
MongoDB 审计基类

基于 Beanie ODM 的审计基类，包含通用的时间戳字段和自动处理逻辑。
"""

from datetime import datetime
from typing import Optional, List, Any
from beanie import before_event, Insert, Update
from pydantic import Field, BaseModel
from common_utils.datetime_utils import get_now_with_timezone


class AuditBase(BaseModel):
    """
    审计基类

    包含通用的时间戳字段和自动处理逻辑

    注意：
    - 单条插入时，@before_event(Insert) 会自动触发设置时间
    - 批量插入时，DocumentBase.insert_many 会委托给本类的 prepare_for_insert_many 处理
    """

    # 系统字段
    created_at: Optional[datetime] = Field(default=None, description="创建时间")
    updated_at: Optional[datetime] = Field(default=None, description="更新时间")

    @before_event(Insert)
    async def set_created_at(self):
        """插入前设置创建时间"""
        now = get_now_with_timezone()
        self.created_at = now
        self.updated_at = now

    @before_event(Update)
    async def set_updated_at(self):
        """更新前设置更新时间"""
        self.updated_at = get_now_with_timezone()

    @classmethod
    def prepare_for_insert_many(cls, documents: List[Any]) -> None:
        """
        批量插入前准备：为文档设置审计时间字段

        由于 Beanie 的 @before_event(Insert) 在批量插入时不会自动触发，
        此方法会被 DocumentBase.insert_many 调用，负责在批量插入前设置审计字段。

        Args:
            documents: 待插入的文档列表

        Note:
            此方法会被 DocumentBase.insert_many 自动调用，
            开发者通常不需要手动调用此方法。
        """
        now = get_now_with_timezone()
        for doc in documents:
            # 只为 None 值的审计字段设置时间，避免覆盖已有值
            if hasattr(doc, 'created_at') and doc.created_at is None:
                doc.created_at = now
            if hasattr(doc, 'updated_at') and doc.updated_at is None:
                doc.updated_at = now


__all__ = ["AuditBase"]
