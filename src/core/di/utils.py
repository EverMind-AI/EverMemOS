# -*- coding: utf-8 -*-
"""
依赖注入工具函数
"""

from typing import Type, TypeVar, List, Dict, Any, Optional, Callable
import inspect

from core.di.container import get_container
from core.di.bean_definition import BeanScope
from core.di.exceptions import BeanNotFoundError

T = TypeVar('T')


def get_bean(name: str) -> Any:
    """
    根据名称获取Bean

    Args:
        name: Bean名称

    Returns:
        Bean实例

    Raises:
        BeanNotFoundError: 当Bean不存在时
    """
    return get_container().get_bean(name)


def get_beans() -> Dict[str, Any]:
    """
    获取所有Bean

    Returns:
        所有Bean的字典，key为name，value为实例
    """
    return get_container().get_beans()


def get_bean_by_type(bean_type: Type[T]) -> T:
    """
    根据类型获取Bean（Primary实现或唯一实现）

    Args:
        bean_type: Bean类型

    Returns:
        Bean实例

    Raises:
        BeanNotFoundError: 当Bean不存在时
    """
    return get_container().get_bean_by_type(bean_type)


def get_beans_by_type(bean_type: Type[T]) -> List[T]:
    """
    根据类型获取所有Bean实现

    Args:
        bean_type: Bean类型

    Returns:
        Bean实例列表
    """
    return get_container().get_beans_by_type(bean_type)


def register_bean(
    bean_type: Type[T],
    instance: T = None,
    name: str = None,
    scope: BeanScope = BeanScope.SINGLETON,
    is_primary: bool = False,
    is_mock: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    注册Bean

    Args:
        bean_type: Bean类型
        instance: Bean实例（可选，如果不提供则会自动创建）
        name: Bean名称
        scope: Bean作用域
        is_primary: 是否为Primary实现
        is_mock: 是否为Mock实现
        metadata: Bean的元数据，可用于存储额外信息
    """
    get_container().register_bean(
        bean_type=bean_type,
        instance=instance,
        bean_name=name,
        scope=scope,
        is_primary=is_primary,
        is_mock=is_mock,
        metadata=metadata,
    )


def register_factory(
    bean_type: Type[T],
    factory_method: Callable[[], T],
    name: str = None,
    is_primary: bool = False,
    is_mock: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    注册Factory方法

    Args:
        bean_type: Bean类型
        factory_method: Factory方法
        name: Bean名称
        is_primary: 是否为Primary实现
        is_mock: 是否为Mock实现
        metadata: Bean的元数据，可用于存储额外信息
    """
    get_container().register_factory(
        bean_type=bean_type,
        factory_method=factory_method,
        bean_name=name,
        is_primary=is_primary,
        is_mock=is_mock,
        metadata=metadata,
    )


def register_singleton(
    bean_type: Type[T], instance: T = None, name: str = None
) -> None:
    """
    注册单例Bean

    Args:
        bean_type: Bean类型
        instance: Bean实例
        name: Bean名称
    """
    register_bean(bean_type, instance, name, BeanScope.SINGLETON)


def register_prototype(bean_type: Type[T], name: str = None) -> None:
    """
    注册原型Bean（每次获取都创建新实例）

    Args:
        bean_type: Bean类型
        name: Bean名称
    """
    register_bean(bean_type, None, name, BeanScope.PROTOTYPE)


def register_primary(bean_type: Type[T], instance: T = None, name: str = None) -> None:
    """
    注册Primary Bean

    Args:
        bean_type: Bean类型
        instance: Bean实例
        name: Bean名称
    """
    register_bean(bean_type, instance, name, BeanScope.SINGLETON, is_primary=True)


def register_mock(bean_type: Type[T], instance: T = None, name: str = None) -> None:
    """
    注册Mock Bean

    Args:
        bean_type: Bean类型
        instance: Bean实例
        name: Bean名称
    """
    register_bean(bean_type, instance, name, BeanScope.SINGLETON, is_mock=True)


def contains_bean(name: str) -> bool:
    """
    检查是否包含指定名称的Bean

    Args:
        name: Bean名称

    Returns:
        是否包含Bean
    """
    return get_container().contains_bean(name)


def contains_bean_by_type(bean_type: Type) -> bool:
    """
    检查是否包含指定类型的Bean

    Args:
        bean_type: Bean类型

    Returns:
        是否包含Bean
    """
    return get_container().contains_bean_by_type(bean_type)


def enable_mock_mode() -> None:
    """启用Mock模式"""
    get_container().enable_mock_mode()


def disable_mock_mode() -> None:
    """禁用Mock模式"""
    get_container().disable_mock_mode()


def is_mock_mode() -> bool:
    """检查是否为Mock模式"""
    return get_container().is_mock_mode()


def clear_container() -> None:
    """清空容器"""
    get_container().clear()


def inject(target_func: Callable) -> Callable:
    """
    函数依赖注入装饰器

    将函数参数按类型自动注入Bean
    """

    def wrapper(*args, **kwargs):
        # 获取函数签名
        signature = inspect.signature(target_func)

        # 准备注入参数
        injected_kwargs = {}
        for param_name, param in signature.parameters.items():
            if param_name not in kwargs and param.annotation != inspect.Parameter.empty:
                try:
                    injected_kwargs[param_name] = get_bean_by_type(param.annotation)
                except BeanNotFoundError:
                    # 如果找不到Bean且参数有默认值，使用默认值
                    if param.default != inspect.Parameter.empty:
                        injected_kwargs[param_name] = param.default
                    else:
                        # 必需参数但找不到Bean，抛出异常
                        raise

        # 合并参数
        kwargs.update(injected_kwargs)
        return target_func(*args, **kwargs)

    return wrapper


def lazy_inject(bean_type: Type[T]) -> Callable[[], T]:
    """
    延迟注入函数

    返回一个lambda函数，调用时才获取Bean

    Args:
        bean_type: Bean类型

    Returns:
        延迟获取Bean的函数
    """
    return lambda: get_bean_by_type(bean_type)


def get_or_create(bean_type: Type[T], factory: Callable[[], T] = None) -> T:
    """
    获取Bean，如果不存在则创建

    Args:
        bean_type: Bean类型
        factory: Factory方法（可选）

    Returns:
        Bean实例
    """
    try:
        return get_bean_by_type(bean_type)
    except BeanNotFoundError:
        if factory:
            instance = factory()
            register_bean(bean_type, instance)
            return instance
        else:
            # 尝试自动创建
            try:
                instance = bean_type()
                register_bean(bean_type, instance)
                return instance
            except Exception as e:
                raise BeanNotFoundError(bean_type=bean_type)


def conditional_register(
    condition: Callable[[], bool],
    bean_type: Type[T],
    instance: T = None,
    name: str = None,
) -> None:
    """
    条件注册Bean

    Args:
        condition: 条件函数
        bean_type: Bean类型
        instance: Bean实例
        name: Bean名称
    """
    if condition():
        register_bean(bean_type, instance, name)


def batch_register(beans: Dict[Type, Any]) -> None:
    """
    批量注册Bean

    Args:
        beans: Bean字典，key为类型，value为实例
    """
    for bean_type, instance in beans.items():
        register_bean(bean_type, instance)


def get_bean_info(bean_type: Type = None, bean_name: str = None) -> Dict[str, Any]:
    """
    获取Bean信息

    Args:
        bean_type: Bean类型
        bean_name: Bean名称

    Returns:
        Bean信息字典
    """
    container = get_container()
    info = {}

    if bean_name:
        if container.contains_bean(bean_name):
            bean_def = container._named_beans[bean_name]
            info = {
                'name': bean_def.bean_name,
                'type': bean_def.bean_type.__name__,
                'scope': bean_def.scope.value,
                'is_primary': bean_def.is_primary,
                'is_mock': bean_def.is_mock,
                'has_instance': bean_def in container._singleton_instances,
            }
    elif bean_type:
        if container.contains_bean_by_type(bean_type):
            definitions = container._bean_definitions[bean_type]
            info = {
                'type': bean_type.__name__,
                'implementations': [
                    {
                        'name': def_.bean_name,
                        'scope': def_.scope.value,
                        'is_primary': def_.is_primary,
                        'is_mock': def_.is_mock,
                    }
                    for def_ in definitions
                ],
            }

    return info


def get_all_beans_info() -> List[Dict[str, Any]]:
    """
    获取所有已注册的Bean信息（结构化数据）

    Returns:
        Bean信息结构化数据列表
    """
    return get_container().list_all_beans_info()


def list_all_beans() -> List[str]:
    """
    列出所有已注册的Bean信息（格式化字符串）

    Returns:
        格式化的Bean信息字符串列表
    """
    beans_info = get_all_beans_info()

    formatted_beans = []
    for bean_info in beans_info:
        flags = []
        if bean_info['is_primary']:
            flags.append("primary")
        if bean_info['is_mock']:
            flags.append("mock")
        flag_str = f" ({', '.join(flags)})" if flags else ""

        formatted_beans.append(
            f"   • {bean_info['name']} ({bean_info['type_name']}) [{bean_info['scope']}]{flag_str}"
        )

    return formatted_beans


def print_container_info():
    """打印容器信息"""
    formatted_beans = list_all_beans()
    from core.observation.logger import info  # 便捷用法，适合偶尔调用

    info(f"\n📦 依赖注入容器信息:")
    info(f"   总Bean数量: {len(formatted_beans)}")
    info(f"   Mock模式: {'启用' if is_mock_mode() else '禁用'}")

    if formatted_beans:
        info("\n📋 已注册的Bean:")
        for bean_line in formatted_beans:
            info(bean_line)
    else:
        info("   无已注册的Bean")
    info("")


# ===============================================

# subclasses


def get_all_subclasses(base_class: Type[T]) -> List[Type[T]]:
    """
    递归获取指定类的所有子类（包括子类的子类）

    Args:
        base_class: 基类

    Returns:
        List[Type[T]]: 所有子类的列表，包括直接子类和间接子类
    """
    subclasses = []
    for subclass in base_class.__subclasses__():
        if subclass != base_class:
            subclasses.append(subclass)
            # 递归获取子类的子类
            subclasses.extend(get_all_subclasses(subclass))
    return subclasses
