# -*- coding: utf-8 -*-
"""
扫描上下文管理器
用于在模块扫描和导入过程中传递上下文信息
"""

from typing import Dict, Any, Optional
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path


# 使用 ContextVar 来存储当前的扫描上下文，支持多线程/异步环境
_current_scan_context: ContextVar[Optional['ScanContext']] = ContextVar(
    'current_scan_context', default=None
)


class ScanContext:
    """
    扫描上下文
    包含模块扫描时的各种上下文信息
    """

    def __init__(
        self,
        file_path: Path,
        module_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化扫描上下文

        Args:
            file_path: 被扫描的文件路径
            module_name: 模块名称
            metadata: 额外的元数据信息
        """
        self.file_path = file_path
        self.module_name = module_name
        self.metadata = metadata or {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取元数据

        Args:
            key: 元数据键
            default: 默认值

        Returns:
            元数据值
        """
        return self.metadata.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        设置元数据

        Args:
            key: 元数据键
            value: 元数据值
        """
        self.metadata[key] = value

    def __repr__(self) -> str:
        return f"ScanContext(module={self.module_name}, file={self.file_path}, metadata={self.metadata})"


class ScanContextRegistry:
    """
    扫描上下文注册器
    用于管理路径到上下文元数据的映射
    """

    def __init__(self):
        """初始化上下文注册器"""
        # 路径到元数据的映射
        self._path_context_mapping: Dict[str, Dict[str, Any]] = {}

    def register(self, path: str, metadata: Dict[str, Any]) -> 'ScanContextRegistry':
        """
        注册扫描路径的上下文元数据

        Args:
            path: 扫描路径
            metadata: 上下文元数据

        Returns:
            self，支持链式调用
        """
        self._path_context_mapping[path] = metadata
        return self

    def unregister(self, path: str) -> 'ScanContextRegistry':
        """
        取消注册扫描路径

        Args:
            path: 扫描路径

        Returns:
            self，支持链式调用
        """
        self._path_context_mapping.pop(path, None)
        return self

    def get_metadata_for_path(self, file_path: Path) -> Dict[str, Any]:
        """
        根据文件路径获取对应的上下文元数据

        Args:
            file_path: 文件路径

        Returns:
            上下文元数据字典
        """
        # 查找最匹配的路径（最长前缀匹配）
        file_path_str = str(file_path.resolve())
        matched_metadata = {}
        max_match_length = 0

        for registered_path, metadata in self._path_context_mapping.items():
            registered_path_resolved = str(Path(registered_path).resolve())
            # 检查文件是否在注册的路径下
            if file_path_str.startswith(registered_path_resolved):
                match_length = len(registered_path_resolved)
                if match_length > max_match_length:
                    max_match_length = match_length
                    matched_metadata = metadata

        return matched_metadata.copy()

    def clear(self) -> 'ScanContextRegistry':
        """
        清空所有注册的路径上下文

        Returns:
            self，支持链式调用
        """
        self._path_context_mapping.clear()
        return self

    def get_all_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有路径到上下文的映射

        Returns:
            路径到上下文元数据的映射字典
        """
        return self._path_context_mapping.copy()


@contextmanager
def scan_context(
    file_path: Path, module_name: str, metadata: Optional[Dict[str, Any]] = None
):
    """
    扫描上下文管理器

    使用方法：
    ```python
    with scan_context(file_path, module_name, {"key": "value"}):
        # 在这个上下文中导入模块
        importlib.import_module(module_name)
        # 被导入的模块可以通过 get_current_scan_context() 获取上下文
    ```

    Args:
        file_path: 被扫描的文件路径
        module_name: 模块名称
        metadata: 额外的元数据
    """
    # 创建上下文对象
    context = ScanContext(file_path, module_name, metadata)

    # 设置当前上下文
    token = _current_scan_context.set(context)

    try:
        yield context
    finally:
        # 恢复之前的上下文
        _current_scan_context.reset(token)


def get_current_scan_context() -> Optional[ScanContext]:
    """
    获取当前的扫描上下文

    在被扫描的模块中调用此函数可以获取当前的扫描上下文

    Returns:
        当前的扫描上下文，如果不在扫描上下文中则返回 None

    Example:
        ```python
        # 在被扫描的模块中
        from core.di.scan_context import get_current_scan_context

        context = get_current_scan_context()
        if context:
            print(f"当前模块: {context.module_name}")
            print(f"文件路径: {context.file_path}")
            print(f"自定义数据: {context.get('custom_key')}")
        ```
    """
    return _current_scan_context.get()


def has_scan_context() -> bool:
    """
    检查是否在扫描上下文中

    Returns:
        如果当前在扫描上下文中返回 True，否则返回 False
    """
    return _current_scan_context.get() is not None
