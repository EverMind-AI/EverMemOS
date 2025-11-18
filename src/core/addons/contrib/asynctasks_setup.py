# -*- coding: utf-8 -*-
"""
异步任务设置模块

处理从 addon 中加载异步任务扫描路径并注册任务的入口函数
"""

from core.asynctasks.task_scan_registry import TaskScanDirectoriesRegistry
from core.asynctasks.task_manager import TaskManager
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger
from core.addons.addons_registry import ADDONS_REGISTRY

logger = get_logger(__name__)


def setup_async_tasks(addons: list = None):
    """
    设置异步任务

    从 addon 列表中提取异步任务扫描目录，并执行任务扫描和注册

    Args:
        addons (list, optional): addon 列表。如果为None，从 ADDONS_REGISTRY 中获取
    """
    logger.info("🔄 正在注册异步任务...")

    try:
        # 获取任务管理器
        task_manager = get_bean_by_type(TaskManager)

        # 如果没有提供 addons，从 ADDONS_REGISTRY 获取
        if addons is None:
            addons = ADDONS_REGISTRY.get_all()

        logger.info("  📦 从 %d 个 addon 中加载异步任务扫描路径...", len(addons))

        # 创建任务目录注册器并从 addons 中填充
        task_directories_registry = TaskScanDirectoriesRegistry()
        for addon in addons:
            if addon.has_asynctasks():
                addon_dirs = addon.asynctasks.get_scan_directories()
                for directory in addon_dirs:
                    task_directories_registry.add_scan_path(directory)
                logger.debug(
                    "  📌 addon [%s] 贡献 %d 个任务目录", addon.name, len(addon_dirs)
                )

        task_directories = task_directories_registry.get_scan_directories()
        logger.info("📂 任务目录数量: %d", len(task_directories))
        for directory in task_directories:
            logger.debug("  + %s", directory)

        # 自动扫描并注册任务
        task_manager.scan_and_register_tasks(task_directories_registry)

        # 打印已注册的任务
        registered_tasks = task_manager.list_registered_task_names()
        logger.info("📋 已注册的任务列表: %s", registered_tasks)

        logger.info("✅ 异步任务注册完成")
    except Exception as e:
        logger.error("❌ 异步任务注册失败: %s", e)
        raise


def print_registered_tasks():
    """打印已注册的异步任务"""
    logger.info("\n📋 已注册的任务列表:")
    logger.info("-" * 50)

    task_manager = get_bean_by_type(TaskManager)

    registered_tasks = task_manager.list_registered_task_names()
    logger.info("📋 已注册的任务列表: %s", registered_tasks)
