"""
应用启动模块

负责应用启动时的各种初始化操作
"""

# 导入依赖注入相关模块
from core.observation.logger import get_logger
from core.addons.addons_registry import ADDONS_REGISTRY
from core.addons.contrib.di_setup import (
    setup_dependency_injection,
    print_registered_beans,
)
from core.addons.contrib.asynctasks_setup import (
    setup_async_tasks,
    print_registered_tasks,
)

# 推荐用法：模块顶部获取一次logger，后续直接使用（高性能）
logger = get_logger(__name__)


def setup_all(load_entrypoints: bool = True):
    """
    设置所有组件

    Args:
        load_entrypoints (bool): 是否从 entry points 加载 addons。默认为 True

    Returns:
        ComponentScanner: 配置好的组件扫描器
    """
    # 0. 加载 addons entry points（如果启用）
    if load_entrypoints:
        logger.info("🔌 正在加载 addons entry points...")
        ADDONS_REGISTRY.load_entrypoints()

    # 获取所有 addons
    all_addons = ADDONS_REGISTRY.get_all()
    logger.info("📦 共加载 %d 个 addon", len(all_addons))

    # 1. 设置依赖注入
    scanner = setup_dependency_injection(all_addons)

    # 2. 设置异步任务
    # setup_async_tasks(all_addons)

    return scanner


if __name__ == "__main__":
    # 启动依赖注入
    setup_all()

    # 打印注册的Bean信息
    print_registered_beans()

    # 打印已注册的任务
    print_registered_tasks()

    logger.info("\n✨ 应用启动完成！")
