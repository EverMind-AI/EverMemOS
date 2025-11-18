# -*- coding: utf-8 -*-
"""
Bean定义模块

包含Bean的定义类和作用域枚举
"""

from enum import Enum
from typing import Type, Callable, Any, Set, Dict, Optional


class BeanScope(str, Enum):
    """Bean作用域枚举"""

    SINGLETON = "singleton"
    PROTOTYPE = "prototype"
    FACTORY = "factory"


class BeanDefinition:
    """Bean定义类"""

    def __init__(
        self,
        bean_type: Type,
        bean_name: str = None,
        scope: BeanScope = BeanScope.SINGLETON,
        is_primary: bool = False,
        is_mock: bool = False,
        factory_method: Callable = None,
        instance: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化Bean定义

        Args:
            bean_type: Bean的类型
            bean_name: Bean的名称，默认为类型名称的小写
            scope: Bean的作用域，默认为单例
            is_primary: 是否为主Bean，用于在有多个实现时优先选择
            is_mock: 是否为Mock实现
            factory_method: 工厂方法，用于创建Bean实例
            instance: 预先创建的实例
            metadata: Bean的元数据，可用于存储额外信息
        """
        self.bean_type = bean_type
        self.bean_name = bean_name or bean_type.__name__.lower()
        self.scope = scope
        self.is_primary = is_primary
        self.is_mock = is_mock
        self.factory_method = factory_method
        self.instance = instance
        self.metadata = metadata or {}
        # 依赖项集合
        self.dependencies: Set[Type] = set()

    def __repr__(self):
        metadata_str = f", metadata={self.metadata}" if self.metadata else ""
        return f"BeanDefinition(type={self.bean_type.__name__}, name={self.bean_name}, scope={self.scope.value}{metadata_str})"
