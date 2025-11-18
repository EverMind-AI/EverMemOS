# -*- coding: utf-8 -*-
"""
Bean排序策略模块

用于根据BeanDefinition的属性（如is_primary、metadata等）来确定Bean的优先级顺序

优先级排序规则（从高到低）：
1. is_mock: Mock模式下，Mock Bean > 非Mock Bean；非Mock模式下，Mock Bean被直接过滤掉
2. 匹配方式: 直接匹配 > 实现类匹配
3. primary: Primary Bean > 非Primary Bean
4. scope: Factory Bean > Regular Bean
"""

from typing import List, Tuple, Set, Type
from core.di.bean_definition import BeanDefinition, BeanScope


class BeanOrderStrategy:
    """
    Bean排序策略类

    根据BeanDefinition的各种属性来计算和排序Bean的优先级
    用于在有多个候选Bean时确定使用顺序

    排序键格式: (mock_priority, match_priority, primary_priority, scope_priority)
    数值越小，优先级越高
    """

    @staticmethod
    def calculate_order_key(
        bean_def: BeanDefinition, is_direct_match: bool, mock_mode: bool = False
    ) -> Tuple[int, int, int, int]:
        """
        计算Bean的排序键

        Args:
            bean_def: Bean定义对象
            is_direct_match: 是否为直接匹配（True=直接匹配，False=实现类匹配）
            mock_mode: 是否处于Mock模式

        Returns:
            Tuple[int, int, int, int]: 排序键元组
            格式: (mock_priority, match_priority, primary_priority, scope_priority)

        优先级规则:
            - mock_priority: Mock模式下，Mock Bean=0, 非Mock Bean=1；非Mock模式下都为0
            - match_priority: 直接匹配=0, 实现类匹配=1
            - primary_priority: Primary Bean=0, 非Primary Bean=1
            - scope_priority: Factory Bean=0, 非Factory Bean=1
        """
        # 1. Mock优先级（仅在Mock模式下区分）
        if mock_mode:
            mock_priority = 0 if bean_def.is_mock else 1
        else:
            mock_priority = 0  # 非Mock模式下不区分

        # 2. 匹配方式优先级（直接匹配优先）
        match_priority = 0 if is_direct_match else 1

        # 3. Primary优先级（Primary优先）
        primary_priority = 0 if bean_def.is_primary else 1

        # 4. Scope优先级（Factory优先）
        scope_priority = 0 if bean_def.scope == BeanScope.FACTORY else 1

        return (mock_priority, match_priority, primary_priority, scope_priority)

    @staticmethod
    def sort_beans_with_context(
        bean_defs: List[BeanDefinition],
        direct_match_types: Set[Type],
        mock_mode: bool = False,
    ) -> List[BeanDefinition]:
        """
        根据上下文信息对Bean定义列表进行排序

        Args:
            bean_defs: Bean定义列表
            direct_match_types: 直接匹配的类型集合
            mock_mode: 是否处于Mock模式

        Returns:
            List[BeanDefinition]: 排序后的Bean定义列表

        注意:
            - 在非Mock模式下，Mock Bean会被直接过滤掉，不参与排序
            - 在Mock模式下，Mock Bean优先于非Mock Bean
        """
        # 在非Mock模式下，过滤掉所有Mock Bean
        if not mock_mode:
            bean_defs = [bd for bd in bean_defs if not bd.is_mock]

        # 为每个Bean计算排序键，然后按键排序
        sorted_beans = sorted(
            bean_defs,
            key=lambda bd: BeanOrderStrategy.calculate_order_key(
                bean_def=bd,
                is_direct_match=bd.bean_type in direct_match_types,
                mock_mode=mock_mode,
            ),
        )
        return sorted_beans

    @staticmethod
    def sort_beans(bean_defs: List[BeanDefinition]) -> List[BeanDefinition]:
        """
        对Bean定义列表进行简单排序（兼容旧接口）

        Args:
            bean_defs: Bean定义列表

        Returns:
            List[BeanDefinition]: 排序后的Bean定义列表

        注意:
            此方法仅考虑primary和scope，不考虑匹配方式和Mock模式
            建议使用 sort_beans_with_context 方法以获得完整的排序功能
        """
        # 按 (primary_priority, scope_priority) 排序
        sorted_beans = sorted(
            bean_defs,
            key=lambda bd: (
                0 if bd.is_primary else 1,  # Primary优先
                0 if bd.scope == BeanScope.FACTORY else 1,  # Factory优先
            ),
        )
        return sorted_beans
