# -*- coding: utf-8 -*-
"""
依赖注入设置模块

处理从 addon 中加载依赖注入扫描路径的入口函数
"""

from core.di.scanner import ComponentScanner
from core.di.utils import get_beans
from core.observation.logger import get_logger
from core.addons.addons_registry import ADDONS_REGISTRY

logger = get_logger(__name__)


def setup_dependency_injection(addons: list = None):
    """
    设置依赖注入框架

    从 addon 列表中提取 DI 扫描路径，并执行组件扫描和注册

    Args:
        addons (list, optional): addon 列表。如果为None，从 ADDONS_REGISTRY 中获取

    Returns:
        ComponentScanner: 配置好的组件扫描器
    """
    logger.info("🚀 正在初始化依赖注入容器...")

    # 导入以触发自动替换Bean排序策略（模块加载时执行）
    from core.addons.contrib import addon_bean_order_strategy  # noqa: F401

    # 创建组件扫描器
    scanner = ComponentScanner()

    # 如果没有提供 addons，从 ADDONS_REGISTRY 获取
    if addons is None:
        addons = ADDONS_REGISTRY.get_all()

    logger.info("  📦 从 %d 个 addon 中加载依赖注入扫描路径...", len(addons))

    # 收集所有扫描路径并注册 scan_context
    total_paths = 0
    for addon in addons:
        if addon.has_di():
            addon_paths = addon.di.get_scan_paths()
            logger.debug(
                "  📌 addon [%s] 贡献 %d 个扫描路径", addon.name, len(addon_paths)
            )

            # 为每个扫描路径注册 scan_context，标记 addon_tag
            for path in addon_paths:
                # 注册扫描上下文，标记来源 addon
                scanner.register_scan_context(path, {"addon_tag": addon.name})
                # 添加扫描路径
                scanner.add_scan_path(path)
                logger.debug("    + %s (addon_tag=%s)", path, addon.name)
                total_paths += 1

    # 执行扫描和注册
    scanner.scan()
    logger.info("✅ 依赖注入设置完成，共扫描 %d 个路径", total_paths)

    return scanner


def print_registered_beans():
    """打印所有已注册的Bean"""
    logger.info("\n📋 已注册的Bean列表:")
    logger.info("-" * 50)

    all_beans = get_beans()
    for name, bean in all_beans.items():
        logger.info("  • %s: %s", name, type(bean).__name__)

    logger.info("\n📊 总计: %d 个Bean", len(all_beans))
