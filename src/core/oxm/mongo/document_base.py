"""
MongoDB 文档基类

基于 Beanie ODM 的文档基类，提供通用的文档基础功能。
"""

from datetime import datetime
from common_utils.datetime_utils import to_timezone
from beanie import Document, WriteRules
from pydantic import model_validator, BaseModel
from typing import Self, List, Optional
from motor.motor_asyncio import AsyncIOMotorClientSession
from pymongo.results import InsertManyResult

from core.oxm.mongo.audit_base import AuditBase

MAX_RECURSION_DEPTH = 4
DEFAULT_DATABASE = "default"


class DocumentBase(Document):
    """
    文档基类

    基于 Beanie Document 的基础文档类，提供通用的文档基础功能
    """

    @classmethod
    def get_bind_database(cls) -> str | None:
        """
        读取绑定数据库名（只读）。

        仅从 `Settings.bind_database` 读取，不提供任何运行时修改入口。
        子类可通过覆写内部 `Settings` 的类变量 `bind_database` 进行绑定：

        class MyDoc(DocumentBase):
            class Settings:
                bind_database = "my_db"
        """
        settings = getattr(cls, "Settings", None)
        if settings is not None:
            return getattr(settings, "bind_database", DEFAULT_DATABASE)
        return DEFAULT_DATABASE

    def _recursive_datetime_check(self, obj, path: str = "", depth: int = 0):
        """
        递归检查并转换所有datetime对象到上海时区

        Args:
            obj: 要检查的对象
            path: 当前对象的路径（用于调试）
            depth: 当前递归深度

        Returns:
            转换后的对象
        """
        # 控制最大递归深度
        if depth >= MAX_RECURSION_DEPTH:
            return obj

        # 情况一：对象是datetime
        if isinstance(obj, datetime):
            if obj.tzinfo is None:
                # 没有时区信息，转换为默认时区；一般是进程里面创建的放到参数里面
                return to_timezone(obj)
            else:
                # 读取的时候带时区且是默认时区（shanghai）返回
                return obj

        # 情况二：对象是BaseModel
        if isinstance(obj, BaseModel):
            for field_name, value in obj:
                new_path = f"{path}.{field_name}" if path else field_name
                new_value = self._recursive_datetime_check(value, new_path, depth + 1)
                # 使用 __dict__ 直接更新值，避免触发验证器
                obj.__dict__[field_name] = new_value
            return obj

        # 情况三：对象是列表、元组或集合（性能优化）
        if isinstance(obj, (list, tuple, set)):
            # 如果集合为空，直接返回
            if not obj:
                return obj

            # list：只检查第一个元素
            if isinstance(obj, list):
                first_item = obj[0]
                first_checked = self._recursive_datetime_check(
                    first_item, f"{path}[0]", depth + 2
                )

                # 如果第一个元素没有变化，认为整个列表都不需要转换
                if first_checked is first_item:
                    return obj

            # set：取任意一个元素检查（set 本身无序，取第一个即可）
            elif isinstance(obj, set):
                sample_item = next(iter(obj))
                sample_checked = self._recursive_datetime_check(
                    sample_item, f"{path}[sample]", depth + 2
                )

                # 如果抽样元素没有变化，认为整个集合都不需要转换
                if sample_checked is sample_item:
                    return obj

            # tuple：只检查前3个元素
            elif isinstance(obj, tuple):
                # 检查前5个元素（或全部，如果长度小于5）
                check_count = min(3, len(obj))
                need_transform = False

                for idx in range(check_count):
                    item = obj[idx]
                    checked = self._recursive_datetime_check(
                        item, f"{path}[{idx}]", depth + 2
                    )
                    if checked is not item:
                        need_transform = True
                        break

                # 如果前5个元素都不需要转换，认为整个 tuple 都不需要转换
                if not need_transform:
                    return obj

            # 需要处理所有元素
            cls = type(obj)
            return cls(
                self._recursive_datetime_check(item, f"{path}[{i}]", depth + 2)
                for i, item in enumerate(obj)
            )

        # 情况四：对象是字典
        if isinstance(obj, dict):
            return {
                key: self._recursive_datetime_check(
                    value, f"{path}[{repr(key)}]", depth + 2
                )
                for key, value in obj.items()
            }

        return obj

    @model_validator(mode='after')
    def check_datetimes_are_aware(self) -> Self:
        """
        递归遍历模型的所有字段，确保任何 datetime 对象都是 'aware' (包含时区信息).
        最多递归3层以避免潜在的问题。

        Returns:
            Self: 当前对象实例
        """
        for field_name, value in self:
            new_value = self._recursive_datetime_check(value, field_name, depth=0)
            if new_value is not value:  # 只在值发生变化时更新

                # 使用 __dict__ 直接更新值，避免触发验证器
                self.__dict__[field_name] = new_value
        return self

    @classmethod
    async def insert_many(
        cls,
        documents: List["DocumentBase"],
        session: Optional[AsyncIOMotorClientSession] = None,
        link_rule: WriteRules = WriteRules.DO_NOTHING,
        **pymongo_kwargs,
    ) -> InsertManyResult:
        """
        重写批量插入方法，委托审计逻辑给 AuditBase

        作为技术入口，检查模型是否继承了 AuditBase，如果是则委托审计字段处理。
        这样保持了职责内聚：DocumentBase 负责协调，AuditBase 负责审计逻辑。

        Args:
            documents: 待插入的文档列表
            session: 可选的 MongoDB 会话，用于事务支持
            link_rule: 关联文档的写入规则
            **pymongo_kwargs: 其他传递给 PyMongo 的参数

        Returns:
            InsertManyResult: 插入结果，包含 inserted_ids
        """
        # 检查模型是否继承了 AuditBase，如果是则委托审计处理

        if issubclass(cls, AuditBase):
            # 委托给 AuditBase 处理审计字段
            AuditBase.prepare_for_insert_many(documents)

        # 调用父类的 insert_many 方法
        return await super().insert_many(
            documents, session=session, link_rule=link_rule, **pymongo_kwargs
        )

    class Settings:
        """文档设置"""

        # 可以在这里设置通用的文档配置
        # 例如：索引、验证规则等

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}({self.id})"

    def __repr__(self) -> str:
        """开发者表示"""
        return f"{self.__class__.__name__}(id={self.id})"
