# -*- coding: utf-8 -*-
"""
依赖注入装饰器
"""

from typing import Type, TypeVar, Optional, Callable, Any, Dict
from functools import wraps

from core.di.container import get_container
from core.di.bean_definition import BeanScope

T = TypeVar('T')


def component(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    组件装饰器

    Args:
        name: Bean名称
        scope: Bean作用域
        lazy: 是否延迟注册
        primary: 是否为主Bean
        metadata: Bean的元数据，可用于存储额外信息
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_component = True
        cls._di_name = name
        cls._di_scope = scope
        cls._di_lazy = lazy
        cls._di_primary = primary
        cls._di_metadata = metadata

        # 检查是否被标记为跳过（通过conditional装饰器）
        if getattr(cls, '_di_skip', False):
            return cls

        if not lazy:
            # 立即注册
            container = get_container()
            container.register_bean(
                bean_type=cls,
                bean_name=name,
                scope=scope,
                is_primary=primary,
                metadata=metadata,
            )

        return cls

    return decorator


def service(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    服务组件装饰器
    """
    return component(name, scope, lazy, primary, metadata)


def repository(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    存储库组件装饰器
    """
    return component(name, scope, lazy, primary, metadata)


def controller(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    控制器组件装饰器
    """
    return component(name, scope, lazy, primary, metadata)


def injectable(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    lazy: bool = False,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    可注入组件装饰器
    """
    return component(name, scope, lazy, primary, metadata)


def mock_impl(
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    primary: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Mock实现装饰器 - 直接注册Mock Bean，由容器机制决定优先级

    Args:
        name: Bean名称
        scope: Bean作用域
        primary: 是否为主Bean
        metadata: Bean的元数据，可用于存储额外信息
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_mock = True
        cls._di_name = name
        cls._di_scope = scope
        cls._di_component = True  # 标记为组件
        cls._di_metadata = metadata

        # 直接注册Mock实现，保持与其他装饰器的一致性
        container = get_container()
        container.register_bean(
            bean_type=cls,
            bean_name=name,
            scope=scope,
            is_primary=getattr(cls, '_di_primary', False),
            is_mock=True,
            metadata=metadata,
        )

        return cls

    return decorator


def factory(
    bean_type: Type[T] = None,
    name: str = None,
    lazy: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Factory装饰器

    Args:
        bean_type: 要创建的Bean类型
        name: Bean名称
        lazy: 是否延迟注册
        metadata: Bean的元数据，可用于存储额外信息
    """

    def decorator(func: Callable[[], T]) -> Callable[[], T]:
        target_type = bean_type or func.__annotations__.get('return', None)

        if not target_type:
            raise ValueError("Factory装饰器必须指定返回类型")

        func._di_factory = True
        func._di_bean_type = target_type
        func._di_name = name
        func._di_lazy = lazy
        func._di_metadata = metadata

        if not lazy:
            # 立即注册Factory
            container = get_container()
            container.register_factory(
                bean_type=target_type,
                factory_method=func,
                bean_name=name,
                metadata=metadata,
            )

        return func

    return decorator


def prototype(cls: Type[T]) -> Type[T]:
    """
    原型作用域装饰器（每次获取都创建新实例）
    """
    cls._di_scope = BeanScope.PROTOTYPE

    # 如果已经是组件，更新作用域
    if hasattr(cls, '_di_component'):
        container = get_container()
        container.register_bean(
            bean_type=cls,
            bean_name=getattr(cls, '_di_name', None),
            scope=BeanScope.PROTOTYPE,
            is_primary=getattr(cls, '_di_primary', False),
            metadata=getattr(cls, '_di_metadata', None),
        )

    return cls


def conditional(condition: Callable[[], bool]):
    """
    条件装饰器 - 控制Bean的条件注册
    注意：应该在 @component 等装饰器之前使用
    """

    def decorator(cls: Type[T]) -> Type[T]:
        # 设置条件标记，让后续的装饰器（如component）根据此条件决定是否注册
        cls._di_conditional = condition

        # 如果条件不满足，标记为跳过
        if not condition():
            cls._di_skip = True

        return cls

    return decorator


def depends_on(*dependencies: Type):
    """
    依赖装饰器 - 声明Bean的依赖关系
    """

    def decorator(cls: Type[T]) -> Type[T]:
        cls._di_dependencies = dependencies
        return cls

    return decorator
