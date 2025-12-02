#!/usr/bin/env python3
"""
Memsys Backend 管理脚本
提供命令行工具来管理后端应用
"""

import asyncio
from IPython.terminal.embed import embed
from functools import wraps
from typing import Callable
import nest_asyncio

nest_asyncio.apply()

import typer
from typer import Typer

# 添加src目录到Python路径
from import_parent_dir import add_parent_path

add_parent_path(0)

# 创建 Typer 应用
cli = Typer(help="Memsys Backend 管理工具")

# 全局变量存储应用状态
_app_state = None
_initialized = False


def setup_environment_and_app(env_file: str = ".env"):
    """
    设置环境和应用

    Args:
        env_file: 环境变量文件名
    """
    global _initialized
    if _initialized:
        return

    from common_utils.load_env import setup_environment
    from application_startup import setup_all

    # 加载环境变量
    setup_environment(load_env_file_name=env_file, check_env_var="GEMINI_API_KEY")

    setup_all()
    _initialized = True


def with_app_context(func: Callable) -> Callable:
    """
    装饰器：为命令提供FastAPI应用上下文

    Args:
        func: 被装饰的异步函数

    Returns:
        装饰后的函数
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        global _app_state

        from app import app

        # 创建应用上下文
        async with app.router.lifespan_context(app):
            # 设置应用状态
            _app_state = app.state
            try:
                # 执行被装饰的函数
                result = await func(*args, **kwargs)
                return result
            finally:
                # 清理应用状态
                _app_state = None

    return wrapper


def with_full_context_decorator(func: Callable) -> Callable:
    """
    装饰器：使用 ContextManager.run_with_full_context 提供完整上下文

    Args:
        func: 被装饰的异步函数

    Returns:
        装饰后的函数
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        global _app_state

        from app import app
        from core.di.utils import get_bean_by_type
        from core.context.context_manager import ContextManager

        # 创建应用上下文
        async with app.router.lifespan_context(app):
            # 设置应用状态
            _app_state = app.state
            try:
                # 获取 ContextManager 实例
                context_manager = get_bean_by_type(ContextManager)

                # 使用 run_with_full_context 执行函数
                result = await context_manager.run_with_full_context(
                    func, *args, auto_commit=True, auto_inherit_user=True, **kwargs
                )
                return result
            finally:
                # 清理应用状态
                _app_state = None

    return wrapper


def is_cli_command(func: Callable) -> Callable:
    """
    装饰器：标记CLI命令函数

    Args:
        func: 被装饰的函数

    Returns:
        装饰后的函数
    """
    func._is_cli_command = True
    return func


@cli.command()
def shell(
    debug: bool = typer.Option(False, "--debug", help="启用调试模式"),
    env_file: str = typer.Option(".env", "--env-file", help="指定要加载的环境变量文件"),
):
    """
    启动交互式shell，提供应用上下文访问
    """
    setup_environment_and_app(env_file)

    from app import app
    from core.observation.logger import get_logger
    from core.di.utils import get_bean_by_type
    from core.context.context_manager import ContextManager

    logger = get_logger(__name__)

    if debug:
        logger.info("调试模式已启用")

    logger.info("使用环境文件: %s", env_file)

    banner = """
    ========================================
    Memsys Backend Shell
    
    可用变量:
    - app: FastAPI应用实例
    - app_state: 应用状态（如果可用）
    - graphs: LangGraph实例（如果可用）
    - logger: 日志记录器
    
    示例用法:
    >>> logger.info("Hello from shell!")
    >>> app.routes  # 查看所有路由
    >>> graphs  # 查看可用的图实例
    ========================================
    """

    def shell_runner():
        embed(header=banner)

    func = with_app_context(with_full_context_decorator(shell_runner))
    asyncio.run(func())


@cli.command()
def list_commands(
    show_all: bool = typer.Option(False, "--all", help="显示所有命令"),
    env_file: str = typer.Option(".env", "--env-file", help="指定要加载的环境变量文件"),
):
    """
    列出所有可用的CLI命令
    """

    if show_all:
        # 显示所有命令，包括隐藏的
        commands = cli.registered_commands
    else:
        # 只显示可见命令
        commands = [cmd for cmd in cli.registered_commands if not cmd.hidden]

    typer.echo("可用的命令:")
    for cmd in commands:
        help_text = cmd.help if cmd.help else "无描述"
        typer.echo(f"  {cmd.name:<20} {help_text}"),

    typer.echo(f"\n使用环境文件: {env_file}")


@cli.command()
def tenant_init(
    env_file: str = typer.Option(".env", "--env-file", help="指定要加载的环境变量文件")
):
    """
    初始化特定租户的 MongoDB 和 Milvus 数据库

    租户ID通过环境变量 TENANT_SINGLE_TENANT_ID 指定。
    数据库连接配置从默认环境变量获取。

    示例:
        # 设置租户ID环境变量
        export TENANT_SINGLE_TENANT_ID=tenant_001

        # 执行初始化
        python src/manage.py tenant-init

        # 使用自定义环境文件
        python src/manage.py tenant-init --env-file .env.production
    """
    from core.observation.logger import get_logger

    logger = get_logger(__name__)

    # 先设置环境和应用
    logger.info("正在设置环境和应用...")
    setup_environment_and_app(env_file)

    # 导入租户初始化模块
    from core.tenants.init_tenant_all import run_tenant_init

    try:
        # 执行租户初始化（从环境变量读取租户ID）
        success = asyncio.run(run_tenant_init())

        # 根据结果设置退出码
        if success:
            logger.info("✅ 所有数据库初始化成功")
            raise typer.Exit(0)
        else:
            logger.error("❌ 部分或全部数据库初始化失败")
            raise typer.Exit(1)
    except ValueError as e:
        # 捕获租户ID未设置的错误
        logger.error("❌ 错误: %s", str(e))
        raise typer.Exit(1)


if __name__ == '__main__':
    # 运行CLI
    cli()
