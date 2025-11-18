# -*- coding: utf-8 -*-
"""
单个 Addon 的扫描注册器容器
用于承载单个 addon 的 DI 和异步任务 registry
"""

from typing import Optional
from core.di.scan_path_registry import ScannerPathsRegistry
from core.asynctasks.task_scan_registry import TaskScanDirectoriesRegistry


class AddonRegistry:
    """
    单个 Addon 的扫描注册器容器

    用于承载单个 addon 的扫描配置：
    - di: DI 组件扫描路径注册器
    - asynctasks: 异步任务扫描目录注册器

    使用示例:
        # 创建 addon registry
        addon = AddonRegistry(name="my_addon")

        # 注册 DI registry
        di_registry = ScannerPathsRegistry()
        di_registry.add_scan_path("/path/to/components")
        addon.register_di(di_registry)

        # 注册异步任务 registry
        task_registry = TaskScanDirectoriesRegistry()
        task_registry.add_scan_path("/path/to/tasks")
        addon.register_asynctasks(task_registry)
    """

    def __init__(self, name: str):
        """
        初始化 addon registry

        Args:
            name: addon 名称
        """
        self.name: str = name
        self.di: Optional[ScannerPathsRegistry] = None
        self.asynctasks: Optional[TaskScanDirectoriesRegistry] = None

    def register_di(self, registry: ScannerPathsRegistry) -> 'AddonRegistry':
        """
        注册 DI 组件扫描路径注册器

        Args:
            registry: DI 扫描路径注册器

        Returns:
            self: 支持链式调用
        """
        self.di = registry
        return self

    def register_asynctasks(
        self, registry: TaskScanDirectoriesRegistry
    ) -> 'AddonRegistry':
        """
        注册异步任务扫描目录注册器

        Args:
            registry: 异步任务扫描目录注册器

        Returns:
            self: 支持链式调用
        """
        self.asynctasks = registry
        return self

    def has_di(self) -> bool:
        """检查是否已注册 DI registry"""
        return self.di is not None

    def has_asynctasks(self) -> bool:
        """检查是否已注册异步任务 registry"""
        return self.asynctasks is not None
