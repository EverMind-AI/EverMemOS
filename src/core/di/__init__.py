# -*- coding: utf-8 -*-
"""
依赖注入模块

注意：为了向后兼容，本模块重新导出了常用的函数和装饰器。
在新代码中，建议从具体的子模块导入以提高可读性：
- 装饰器：from core.di.decorators import service, repository, component, mock_impl, factory
- 工具函数：from core.di.utils import get_bean_by_type, get_bean, enable_mock_mode, disable_mock_mode
- 容器：from core.di.container import get_container
"""

# 装饰器 (来自 decorators.py)
from core.di.decorators import (
    component,
    service,
    repository,
    controller,
    injectable,
    mock_impl,
    factory,
    prototype,
    conditional,
    depends_on,
)

# 工具函数 (来自 utils.py)
from core.di.utils import (
    get_bean,
    get_beans,
    get_bean_by_type,
    get_beans_by_type,
    register_bean,
    register_factory,
    register_singleton,
    register_prototype,
    register_primary,
    register_mock,
    contains_bean,
    contains_bean_by_type,
    enable_mock_mode,
    disable_mock_mode,
    is_mock_mode,
    clear_container,
    inject,
    lazy_inject,
    get_or_create,
    conditional_register,
    batch_register,
    get_bean_info,
    get_all_beans_info,
    list_all_beans,
    print_container_info,
    get_all_subclasses,
)

# 容器 (来自 container.py)
from core.di.container import get_container

# 定义公开的API
__all__ = [
    # 装饰器
    'component',
    'service',
    'repository',
    'controller',
    'injectable',
    'mock_impl',
    'factory',
    'prototype',
    'conditional',
    'depends_on',
    # 核心工具函数
    'get_bean',
    'get_beans',
    'get_bean_by_type',
    'get_beans_by_type',
    'get_container',
    # 注册函数
    'register_bean',
    'register_factory',
    'register_singleton',
    'register_prototype',
    'register_primary',
    'register_mock',
    # 容器检查
    'contains_bean',
    'contains_bean_by_type',
    # Mock模式
    'enable_mock_mode',
    'disable_mock_mode',
    'is_mock_mode',
    # 其他工具
    'clear_container',
    'inject',
    'lazy_inject',
    'get_or_create',
    'conditional_register',
    'batch_register',
    # 信息查询
    'get_bean_info',
    'get_all_beans_info',
    'list_all_beans',
    'print_container_info',
    # 子类查询
    'get_all_subclasses',
]
