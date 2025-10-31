"""
MongoDB 文档基类

基于 Beanie ODM 的文档基类，提供通用的文档基础功能。
"""

from datetime import datetime
from common_utils.datetime_utils import to_timezone

try:
    from beanie import Document
    from pydantic import model_validator, BaseModel
    from typing import Self

    BEANIE_AVAILABLE = True
except ImportError:
    BEANIE_AVAILABLE = False

MAX_RECURSION_DEPTH = 4
DEFAULT_DATABASE = "default"

if BEANIE_AVAILABLE:

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

            # # 情况二：对象是BaseModel
            if isinstance(obj, BaseModel):
                for field_name, value in obj:
                    new_path = f"{path}.{field_name}" if path else field_name
                    new_value = self._recursive_datetime_check(
                        value, new_path, depth + 1
                    )
                    # 使用 __dict__ 直接更新值，避免触发验证器
                    obj.__dict__[field_name] = new_value
                return obj

            # 情况三：对象是列表、元组或集合
            if isinstance(obj, (list, tuple, set)):
                cls = type(obj)
                return cls(
                    self._recursive_datetime_check(item, f"{path}[{i}]", depth + 1)
                    for i, item in enumerate(obj)
                )

            # 情况四：对象是字典
            if isinstance(obj, dict):
                return {
                    key: self._recursive_datetime_check(
                        value, f"{path}[{repr(key)}]", depth + 1
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

else:
    # 如果 Beanie 不可用，提供一个空的基类
    class DocumentBase:
        """
        文档基类占位符

        当 Beanie 依赖不可用时使用
        """

        def __init__(self):
            raise ImportError(
                "Beanie ODM is not available. Please install beanie to use DocumentBase."
            )


# 导出
__all__ = ["DocumentBase", "BEANIE_AVAILABLE"]
