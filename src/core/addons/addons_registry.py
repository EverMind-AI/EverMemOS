# -*- coding: utf-8 -*-
"""
Addons 注册器管理
管理所有 addon 的注册器列表，提供统一的访问接口
"""

import os
from typing import List, Optional, Set
from core.addons.addon_registry import AddonRegistry
from core.observation.logger import get_logger

logger = get_logger(__name__)


class AddonsRegistry:
    """
    Addons 注册器管理器

    管理所有 addon 的注册器列表，提供统一的注册和访问接口

    使用示例:
        from core.addons.addons_registry import ADDONS_REGISTRY

        # 创建并注册 addon
        addon = AddonRegistry(name="my_addon")
        addon.register_di(di_registry).register_asynctasks(task_registry)

        ADDONS_REGISTRY.register(addon)

        # 获取所有 addon
        all_addons = ADDONS_REGISTRY.get_all()

        # 根据名称查找 addon
        my_addon = ADDONS_REGISTRY.get_by_name("my_addon")

        # 从 entry points 自动加载（推荐）
        ADDONS_REGISTRY.load_entrypoints()

    Entry Points 注册方式:
        在 pyproject.toml 中配置：

        [project.entry-points."memsys.addons"]
        my_addon = "my_package.addon_module"

        模块中执行注册：

        # my_package/addon_module.py
        from core.addons.addons_registry import ADDONS_REGISTRY

        my_addon = AddonRegistry(name="my_addon")
        # ... 配置 addon ...
        ADDONS_REGISTRY.register(my_addon)  # 模块导入时自动执行
    """

    def __init__(self):
        """初始化 addons 注册器管理器"""
        self._addons: List[AddonRegistry] = []

    def register(self, addon: AddonRegistry) -> 'AddonsRegistry':
        """
        注册一个 addon

        Args:
            addon: addon 注册器实例

        Returns:
            self: 支持链式调用
        """
        self._addons.append(addon)
        return self

    def get_all(self) -> List[AddonRegistry]:
        """
        获取所有已注册的 addon

        Returns:
            所有 addon 列表
        """
        return self._addons.copy()

    def get_by_name(self, name: str) -> Optional[AddonRegistry]:
        """
        根据名称获取 addon

        Args:
            name: addon 名称

        Returns:
            找到的 addon，如果不存在则返回 None
        """
        for addon in self._addons:
            if addon.name == name:
                return addon
        return None

    def clear(self) -> 'AddonsRegistry':
        """
        清空所有已注册的 addon

        Returns:
            self: 支持链式调用
        """
        self._addons.clear()
        return self

    def count(self) -> int:
        """
        获取已注册的 addon 数量

        Returns:
            addon 数量
        """
        return len(self._addons)

    def _should_load_entrypoint(self, entrypoint_name: str) -> bool:
        """
        根据环境变量判断是否应该加载指定的 entrypoint

        通过 MEMSYS_ENTRYPOINTS_FILTER 环境变量控制加载哪些 entrypoint
        格式：MEMSYS_ENTRYPOINTS_FILTER=ep1,ep2,ep3

        如果环境变量未设置或为空，则加载所有 entrypoint
        如果设置了环境变量，则只加载列表中指定的 entrypoint

        Args:
            entrypoint_name: entrypoint 的名称（ep.name）

        Returns:
            True 表示应该加载，False 表示应该跳过
        """
        filter_config = os.environ.get('MEMSYS_ENTRYPOINTS_FILTER', '').strip()

        # 如果未设置环境变量或为空，加载所有 entrypoint
        if not filter_config:
            return True

        # 逗号分割并过滤
        allowed_entrypoints: Set[str] = {
            name.strip() for name in filter_config.split(',') if name.strip()
        }
        return entrypoint_name in allowed_entrypoints

    def load_entrypoints(self) -> 'AddonsRegistry':
        """
        从 entry points 自动加载所有已注册的 addons

        扫描 'memsys.addons' entry point group，自动发现并加载所有通过该机制注册的 addon。

        工作原理：
        1. 扫描 pyproject.toml 中 [project.entry-points."memsys.addons"] 下的所有 entry points
        2. 根据 MEMSYS_ENTRYPOINTS_FILTER 环境变量过滤需要加载的 entrypoint（通过 ep.name）
        3. 加载每个 entry point 对应的模块（触发模块导入）
        4. 模块导入时会自动执行模块级代码，包括 ADDONS_REGISTRY.register(addon) 调用
        5. 所有 addon 自动注册到全局 ADDONS_REGISTRY 中

        环境变量控制：
        - MEMSYS_ENTRYPOINTS_FILTER: 控制加载哪些 entrypoint，格式为逗号分隔的 entrypoint 名称列表
          示例: MEMSYS_ENTRYPOINTS_FILTER=ep1,ep2,ep3
          如果未设置或为空，则加载所有 entrypoint
          注意：一个 entrypoint 可能包含多个 addon 的注册

        注意：
        - 不需要 entry point 返回特定对象
        - 只需确保模块导入时执行注册代码即可
        - 避免在模块级代码中执行耗时操作

        Returns:
            self: 支持链式调用
        """
        try:
            # Python 3.10+ 使用 importlib.metadata
            from importlib.metadata import entry_points

            logger.info("🔌 开始加载 addons entry points...")

            # 获取 memsys.addons group 下的所有 entry points
            # Python 3.10+ 使用 select 方法，Python 3.9 使用字典访问
            try:
                # Python 3.10+
                addon_eps = entry_points(group='memsys.addons')
            except TypeError:
                # Python 3.9 fallback
                eps = entry_points()
                if hasattr(eps, 'select'):
                    addon_eps = eps.select(group='memsys.addons')
                else:
                    # 直接字典访问
                    addon_eps = (
                        eps.get('memsys.addons', []) if isinstance(eps, dict) else []
                    )

            for ep in addon_eps:
                try:
                    # 根据环境变量过滤 entrypoint
                    if not self._should_load_entrypoint(ep.name):
                        logger.info(
                            "  ⏭️  跳过 entrypoint: %s (未在 MEMSYS_ENTRYPOINTS_FILTER 中)",
                            ep.name,
                        )
                        continue

                    # 加载 entry point，触发模块导入和执行
                    # 模块导入时会自动执行注册代码（如 ADDONS_REGISTRY.register(addon)）
                    ep.load()
                    logger.info("  ✅ 已加载 entrypoint: %s", ep.name)

                except Exception as e:  # pylint: disable=broad-except
                    logger.error("  ❌ 加载 entrypoint %s 失败: %s", ep.name, e)

            logger.info("✅ Addons entry points 加载完成，共 %d 个", self.count())

        except ImportError:
            logger.warning("⚠️  importlib.metadata 不可用，跳过 entry points 加载")
        except Exception as e:  # pylint: disable=broad-except
            logger.error("❌ 加载 addons entry points 失败: %s", e)

        return self


# 全局单例实例
ADDONS_REGISTRY = AddonsRegistry()
