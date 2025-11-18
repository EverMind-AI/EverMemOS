# -*- coding: utf-8 -*-
"""
依赖注入容器核心实现

锁使用策略：
- 纯读操作（如 is_mock_mode, contains_bean*）：无锁，因为读取不可变属性
- 修改容器状态的操作：使用 self._lock 保护
- 获取Bean的操作：需要锁，因为可能创建和缓存单例实例
- 全局容器创建：使用 _container_lock 保证单例
"""

import inspect
import abc
from typing import (
    Dict,
    Type,
    TypeVar,
    Optional,
    Any,
    List,
    Set,
    Callable,
    Union,
    get_origin,
    get_args,
)
from threading import RLock

from core.di.bean_definition import BeanDefinition, BeanScope
from core.di.bean_order_strategy import BeanOrderStrategy
from core.di.scan_context import get_current_scan_context
from core.di.exceptions import (
    CircularDependencyError,
    BeanNotFoundError,
    DuplicateBeanError,
    FactoryError,
    DependencyResolutionError,
    MockNotEnabledError,
)

T = TypeVar('T')


class DIContainer:
    """依赖注入容器"""

    # 类级别的Bean排序策略，可以被替换
    _bean_order_strategy_class = BeanOrderStrategy

    @classmethod
    def replace_bean_order_strategy(cls, strategy_class):
        """
        替换Bean排序策略类

        Args:
            strategy_class: 新的排序策略类，必须具有与BeanOrderStrategy兼容的接口

        注意:
            这是一个临时方案，因为DI机制还没有完全建立。
            此方法会影响所有DIContainer实例的排序行为。
        """
        cls._bean_order_strategy_class = strategy_class

    def __init__(self):
        self._lock = RLock()
        # 按类型存储Bean定义 {Type: [BeanDefinition]}
        self._bean_definitions: Dict[Type, List[BeanDefinition]] = {}
        # 按名称存储Bean定义 {name: BeanDefinition}
        self._named_beans: Dict[str, BeanDefinition] = {}

        # 存储单例实例 {BeanDefinition: instance}
        self._singleton_instances: Dict[BeanDefinition, Any] = {}

        # Mock模式
        self._mock_mode = False
        # 依赖解析栈，用于检测循环依赖
        self._resolving_stack: List[Type] = []

        # 性能优化缓存
        # 类型继承关系缓存 {parent_type: [child_types]}
        self._inheritance_cache: Dict[Type, List[Type]] = {}
        # 候选Bean缓存 {(Type, mock_mode): [BeanDefinition]}
        self._candidates_cache: Dict[tuple, List[BeanDefinition]] = {}
        # 缓存失效标志
        self._cache_dirty = False

    def enable_mock_mode(self):
        """启用Mock模式"""
        with self._lock:
            if not self._mock_mode:
                self._mock_mode = True
                self._invalidate_cache()

    def disable_mock_mode(self):
        """禁用Mock模式"""
        with self._lock:
            if self._mock_mode:
                self._mock_mode = False
                self._invalidate_cache()

    def is_mock_mode(self) -> bool:
        """检查是否为Mock模式"""
        return self._mock_mode

    def _create_bean_definition(
        self,
        bean_type: Type[T],
        bean_name: str = None,
        scope: BeanScope = BeanScope.SINGLETON,
        is_primary: bool = False,
        is_mock: bool = False,
        factory_method: Callable = None,
        instance: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BeanDefinition:
        """
        创建 BeanDefinition，并自动合并扫描上下文中的 metadata

        Args:
            bean_type: Bean的类型
            bean_name: Bean的名称
            scope: Bean的作用域
            is_primary: 是否为主Bean
            is_mock: 是否为Mock实现
            factory_method: 工厂方法
            instance: 预先创建的实例
            metadata: Bean的元数据

        Returns:
            BeanDefinition 实例
        """
        # 合并 metadata：先从 scan_context 获取，再与传入的 metadata 合并
        merged_metadata = {}

        # 1. 检查是否在扫描上下文中，如果是则获取上下文的 metadata
        scan_context = get_current_scan_context()
        if scan_context:
            # 从扫描上下文中获取 metadata
            context_metadata = scan_context.metadata.copy()
            merged_metadata.update(context_metadata)

        # 2. 合并传入的 metadata（传入的优先级更高，可以覆盖扫描上下文的）
        if metadata:
            merged_metadata.update(metadata)

        # 3. 创建 BeanDefinition
        bean_def = BeanDefinition(
            bean_type=bean_type,
            bean_name=bean_name,
            scope=scope,
            is_primary=is_primary,
            is_mock=is_mock,
            factory_method=factory_method,
            instance=instance,
            metadata=merged_metadata if merged_metadata else None,
        )

        return bean_def

    def register_bean(
        self,
        bean_type: Type[T],
        bean_name: str = None,
        scope: BeanScope = BeanScope.SINGLETON,
        is_primary: bool = False,
        is_mock: bool = False,
        instance: T = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'DIContainer':
        """
        注册Bean

        Args:
            bean_type: Bean的类型
            bean_name: Bean的名称
            scope: Bean的作用域
            is_primary: 是否为主Bean
            is_mock: 是否为Mock实现
            instance: 预先创建的实例
            metadata: Bean的元数据，可用于存储额外信息
        """
        with self._lock:
            # 使用统一的方法创建 BeanDefinition，会自动合并扫描上下文的 metadata
            bean_def = self._create_bean_definition(
                bean_type=bean_type,
                bean_name=bean_name,
                scope=scope,
                is_primary=is_primary,
                is_mock=is_mock,
                instance=instance,
                metadata=metadata,
            )

            # 检查重复注册
            if bean_def.bean_name in self._named_beans:
                existing = self._named_beans[bean_def.bean_name]
                if not (is_mock or existing.is_mock):
                    raise DuplicateBeanError(bean_name=bean_def.bean_name)

            # 注册Bean定义
            if bean_type not in self._bean_definitions:
                self._bean_definitions[bean_type] = []
            self._bean_definitions[bean_type].append(bean_def)
            self._named_beans[bean_def.bean_name] = bean_def

            # 分析依赖关系
            self._analyze_dependencies(bean_def)

            # 如果提供了实例，直接存储
            if instance is not None:
                self._singleton_instances[bean_def] = instance

            # 使缓存失效
            self._invalidate_cache()

            return self

    def register_factory(
        self,
        bean_type: Type[T],
        factory_method: Callable[[], T],
        bean_name: str = None,
        is_primary: bool = False,
        is_mock: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'DIContainer':
        """
        注册Factory方法

        Args:
            bean_type: Bean的类型
            factory_method: 工厂方法
            bean_name: Bean的名称
            is_primary: 是否为主Bean
            is_mock: 是否为Mock实现
            metadata: Bean的元数据，可用于存储额外信息
        """
        with self._lock:
            # 使用统一的方法创建 BeanDefinition，会自动合并扫描上下文的 metadata
            bean_def = self._create_bean_definition(
                bean_type=bean_type,
                bean_name=bean_name,
                scope=BeanScope.FACTORY,
                is_primary=is_primary,
                is_mock=is_mock,
                factory_method=factory_method,
                metadata=metadata,
            )

            # 检查重复注册
            if bean_def.bean_name in self._named_beans:
                existing = self._named_beans[bean_def.bean_name]
                if not (is_mock or existing.is_mock):
                    raise DuplicateBeanError(bean_name=bean_def.bean_name)

            # 注册Bean定义
            if bean_type not in self._bean_definitions:
                self._bean_definitions[bean_type] = []
            self._bean_definitions[bean_type].append(bean_def)
            self._named_beans[bean_def.bean_name] = bean_def

            # 使缓存失效
            self._invalidate_cache()

            return self

    def get_bean(self, bean_name: str) -> Any:
        """根据名称获取Bean"""
        with self._lock:
            if bean_name not in self._named_beans:
                raise BeanNotFoundError(bean_name=bean_name)

            bean_def = self._named_beans[bean_name]
            return self._create_instance(bean_def)

    def get_bean_by_type(self, bean_type: Type[T]) -> T:
        """根据类型获取Bean（返回Primary或唯一实现）"""
        with self._lock:
            candidates = self._get_candidates_with_priority(bean_type)

            if not candidates:
                raise BeanNotFoundError(bean_type=bean_type)

            # 如果只有一个候选者，返回它
            if len(candidates) == 1:
                return self._create_instance(candidates[0])

            # 多个候选者，返回优先级最高的
            return self._create_instance(candidates[0])

    def _get_candidates_with_priority(self, bean_type: Type) -> List[BeanDefinition]:
        """
        获取类型的候选Bean定义（按优先级排序）

        优先级排序规则（从高到低）：
        1. is_mock: Mock Bean > 非Mock Bean（仅在Mock模式下生效）
        2. 匹配方式: 直接匹配 > 实现类匹配
        3. primary: Primary Bean > 非Primary Bean
        4. scope: Factory Bean > Regular Bean
        """
        # 使用缓存键
        cache_key = (bean_type, self._mock_mode)

        # 检查缓存
        if cache_key in self._candidates_cache:
            return self._candidates_cache[cache_key]

        # 确保继承关系缓存是最新的
        self._build_inheritance_cache()

        # 收集所有候选Bean
        all_candidates = []
        direct_match_types = set()

        # 1. 收集直接匹配的Bean（包括Primary和非Primary）
        if bean_type in self._bean_definitions:
            for bean_def in self._bean_definitions[bean_type]:
                if self._is_bean_available(bean_def):
                    all_candidates.append(bean_def)
                    direct_match_types.add(bean_def.bean_type)

        # 2. 收集实现类匹配的Bean（接口/抽象类的实现）
        impl_types = self._inheritance_cache.get(bean_type, [])
        for impl_type in impl_types:
            if impl_type in self._bean_definitions:
                for bean_def in self._bean_definitions[impl_type]:
                    if self._is_bean_available(bean_def):
                        all_candidates.append(bean_def)
                        # impl_type 不加入 direct_match_types，因为它是实现类匹配

        # 3. 使用当前配置的Bean排序策略进行统一排序
        priority_candidates = self._bean_order_strategy_class.sort_beans_with_context(
            bean_defs=all_candidates,
            direct_match_types=direct_match_types,
            mock_mode=self._mock_mode,
        )

        # 缓存结果
        self._candidates_cache[cache_key] = priority_candidates
        return priority_candidates

    def get_beans_by_type(self, bean_type: Type[T]) -> List[T]:
        """根据类型获取所有Bean实现"""
        with self._lock:
            candidates = self._get_candidates_with_priority(bean_type)
            return [self._create_instance(bean_def) for bean_def in candidates]

    def get_beans(self) -> Dict[str, Any]:
        """获取所有已注册的Bean"""
        with self._lock:
            result = {}
            for name, bean_def in self._named_beans.items():
                if self._is_bean_available(bean_def):
                    try:
                        result[name] = self._create_instance(bean_def)
                    except Exception:
                        # 跳过无法创建的Bean
                        continue
            return result

    def contains_bean(self, bean_name: str) -> bool:
        """检查是否包含指定名称的Bean"""
        return bean_name in self._named_beans

    def contains_bean_by_type(self, bean_type: Type) -> bool:
        """检查是否包含指定类型的Bean"""
        return bean_type in self._bean_definitions

    def clear(self):
        """清空容器"""
        with self._lock:
            self._bean_definitions.clear()
            self._named_beans.clear()
            self._singleton_instances.clear()
            self._resolving_stack.clear()
            self._invalidate_cache()

    def list_all_beans_info(self) -> List[Dict[str, Any]]:
        """
        列出所有已注册的Bean信息

        Returns:
            Bean信息列表，每个Bean包含：
            - name: Bean名称
            - type_name: Bean类型名称
            - scope: Bean作用域
            - is_primary: 是否为Primary Bean
            - is_mock: 是否为Mock Bean
        """
        beans_info = []

        # 收集所有Bean信息
        for name, bean_def in self._named_beans.items():
            if self._is_bean_available(bean_def):
                beans_info.append(
                    {
                        'name': name,
                        'type_name': bean_def.bean_type.__name__,
                        'scope': bean_def.scope.value,
                        'is_primary': bean_def.is_primary,
                        'is_mock': bean_def.is_mock,
                    }
                )

        return beans_info

    def _invalidate_cache(self):
        """使所有缓存失效"""
        self._inheritance_cache.clear()
        self._candidates_cache.clear()
        self._cache_dirty = True

    def _is_bean_available(self, bean_def: BeanDefinition) -> bool:
        """检查Bean是否在当前模式下可用"""
        if self._mock_mode:
            # Mock模式下，mock和非mock的bean都可用
            return True
        else:
            # 非Mock模式下，只有非mock的bean可用
            return not bean_def.is_mock

    def _build_inheritance_cache(self):
        """构建类型继承关系缓存"""
        if not self._cache_dirty:
            return

        self._inheritance_cache.clear()

        # 获取已注册的类型
        registered_types = list(self._bean_definitions.keys())

        # 额外收集ABC父类类型（排除abc.ABC基类）
        all_parent_types = set(registered_types)
        for registered_type in registered_types:
            try:
                # 获取所有父类，特别是ABC抽象基类
                for base in registered_type.__mro__[1:]:  # 跳过自身
                    # 排除abc.ABC基类和object基类，它们太通用了
                    if (
                        base != abc.ABC
                        and base != object
                        and hasattr(base, '__abstractmethods__')
                    ):  # ABC类型
                        all_parent_types.add(base)
            except (AttributeError, TypeError):
                # 处理非类型的情况
                continue

        # 为所有类型（包括ABC父类）建立继承关系索引
        # parent_type -> [实现它的子类列表]
        for parent_type in all_parent_types:
            child_implementations = []
            for child_type in registered_types:
                if child_type != parent_type:
                    try:
                        if issubclass(child_type, parent_type):
                            child_implementations.append(child_type)
                    except TypeError:
                        # 处理非类型的情况
                        continue
            if child_implementations:
                self._inheritance_cache[parent_type] = child_implementations

        self._cache_dirty = False

    def _create_instance(self, bean_def: BeanDefinition) -> Any:
        """创建Bean实例"""
        # 检查循环依赖
        if bean_def.bean_type in self._resolving_stack:
            dependency_chain = self._resolving_stack + [bean_def.bean_type]
            raise CircularDependencyError(dependency_chain)

        # 处理不同作用域
        if bean_def.scope == BeanScope.SINGLETON:
            # 单例模式：检查缓存，如果有直接返回
            if bean_def in self._singleton_instances:
                return self._singleton_instances[bean_def]

        elif bean_def.scope == BeanScope.FACTORY:
            # 工厂模式：每次调用工厂方法创建新实例
            if bean_def.factory_method:
                try:
                    return bean_def.factory_method()
                except Exception as e:
                    raise FactoryError(bean_def.bean_type, str(e))
            else:
                raise FactoryError(bean_def.bean_type, "未设置Factory方法")

        elif bean_def.scope == BeanScope.PROTOTYPE:
            # 原型模式：每次都创建新实例，不缓存
            try:
                self._resolving_stack.append(bean_def.bean_type)
                return self._instantiate_with_dependencies(bean_def)
            finally:
                if bean_def.bean_type in self._resolving_stack:
                    self._resolving_stack.remove(bean_def.bean_type)

        # 如果有预设实例，直接返回
        if bean_def.instance is not None:
            return bean_def.instance

        # 创建新实例（SINGLETON 作用域）
        try:
            self._resolving_stack.append(bean_def.bean_type)
            instance = self._instantiate_with_dependencies(bean_def)

            # 存储单例实例
            if bean_def.scope == BeanScope.SINGLETON:
                self._singleton_instances[bean_def] = instance

            return instance
        finally:
            if bean_def.bean_type in self._resolving_stack:
                self._resolving_stack.remove(bean_def.bean_type)

    def _instantiate_with_dependencies(self, bean_def: BeanDefinition) -> Any:
        """实例化Bean并注入依赖"""
        bean_type = bean_def.bean_type

        # 获取构造函数签名
        try:
            signature = inspect.signature(bean_type.__init__)
        except Exception:
            # 如果无法获取签名，尝试无参构造
            return bean_type()

        # 准备构造函数参数
        init_params = {}
        for param_name, param in signature.parameters.items():
            if param_name == 'self':
                continue

            # 尝试根据类型注入依赖
            if param.annotation != inspect.Parameter.empty:
                try:
                    # 检查是否为泛型类型（如 List[T]）
                    origin = get_origin(param.annotation)
                    if origin is list or origin is List:
                        # 处理 List[T] 类型的依赖注入
                        args = get_args(param.annotation)
                        if args:
                            # 获取泛型参数类型
                            element_type = args[0]
                            # 注入该类型的所有实现
                            dependencies = self.get_beans_by_type(element_type)
                            init_params[param_name] = dependencies
                        else:
                            # 如果没有泛型参数，尝试空列表
                            init_params[param_name] = []
                    else:
                        # 普通类型的依赖注入
                        dependency = self.get_bean_by_type(param.annotation)
                        init_params[param_name] = dependency
                except BeanNotFoundError:
                    if param.default == inspect.Parameter.empty:
                        # 必需参数但找不到依赖
                        raise DependencyResolutionError(bean_type, param.annotation)

        return bean_type(**init_params)

    def _analyze_dependencies(self, bean_def: BeanDefinition):
        """分析Bean的依赖关系"""
        try:
            signature = inspect.signature(bean_def.bean_type.__init__)
            for param_name, param in signature.parameters.items():
                if param_name == 'self':
                    continue
                if param.annotation != inspect.Parameter.empty:
                    bean_def.dependencies.add(param.annotation)
        except Exception:
            # 如果无法分析，跳过
            pass


# 全局容器实例
_global_container: Optional[DIContainer] = None
_container_lock = RLock()


def get_container() -> DIContainer:
    """获取全局容器实例"""
    global _global_container
    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = DIContainer()
    return _global_container
