# -*- coding: utf-8 -*-
"""
Addon Bean排序策略模块

扩展标准的BeanOrderStrategy，增加对addon_tag的优先级支持

优先级排序规则（从高到低）：
1. addon_tag: 根据环境变量配置的addon优先级（数值越小优先级越高）
2. is_mock: Mock模式下，Mock Bean > 非Mock Bean；非Mock模式下，Mock Bean被直接过滤掉
3. 匹配方式: 直接匹配 > 实现类匹配
4. primary: Primary Bean > 非Primary Bean
5. scope: Factory Bean > Regular Bean
"""

import os
from typing import List, Tuple, Set, Type, Dict
from core.di.bean_definition import BeanDefinition, BeanScope
from core.di.bean_order_strategy import BeanOrderStrategy
from core.observation.logger import get_logger

logger = get_logger(__name__)


class AddonBeanOrderStrategy(BeanOrderStrategy):
    """
    Addon Bean排序策略类

    继承自BeanOrderStrategy，扩展支持addon_tag优先级
    addon_tag的优先级通过环境变量 ADDON_PRIORITY 配置
    格式: "addon1:priority1,addon2:priority2"
    例如: "core:1000,enterprise:50" 表示 enterprise 优先级更高（数值越小优先级越高）
    """

    # 默认的addon优先级配置
    DEFAULT_ADDON_PRIORITY = "core:1000,enterprise:50"

    # addon优先级缓存
    _addon_priority_map: Dict[str, int] = None

    @classmethod
    def _load_addon_priority_map(cls) -> Dict[str, int]:
        """
        从环境变量加载addon优先级配置

        Returns:
            Dict[str, int]: addon名称到优先级的映射（数值越小优先级越高）
        """
        if cls._addon_priority_map is not None:
            return cls._addon_priority_map

        # 从环境变量读取配置，如果没有则使用默认值
        priority_config = os.getenv("ADDON_PRIORITY", cls.DEFAULT_ADDON_PRIORITY)

        priority_map = {}
        for item in priority_config.split(","):
            item = item.strip()
            if ":" in item:
                addon_name, priority_str = item.split(":", 1)
                try:
                    priority_map[addon_name.strip()] = int(priority_str.strip())
                except ValueError:
                    # 忽略无效的配置
                    pass

        cls._addon_priority_map = priority_map
        return priority_map

    @classmethod
    def get_addon_priority(cls, bean_def: BeanDefinition) -> int:
        """
        获取Bean的addon优先级

        Args:
            bean_def: Bean定义对象

        Returns:
            int: addon优先级值，数值越小优先级越高
            如果没有配置addon_tag或未在配置中找到，返回默认值99999（最低优先级）
        """
        priority_map = cls._load_addon_priority_map()

        # 从Bean的metadata中获取addon_tag
        addon_tag = bean_def.metadata.get("addon_tag")
        if not addon_tag:
            # 没有addon_tag，返回最低优先级
            return 99999

        # 返回配置的优先级，如果未配置则返回默认的最低优先级
        return priority_map.get(addon_tag, 99999)

    @staticmethod
    def calculate_order_key(
        bean_def: BeanDefinition, is_direct_match: bool, mock_mode: bool = False
    ) -> Tuple[int, int, int, int, int]:
        """
        计算Bean的排序键（扩展版本，包含addon优先级）

        Args:
            bean_def: Bean定义对象
            is_direct_match: 是否为直接匹配（True=直接匹配，False=实现类匹配）
            mock_mode: 是否处于Mock模式

        Returns:
            Tuple[int, int, int, int, int]: 排序键元组
            格式: (addon_priority, mock_priority, match_priority, primary_priority, scope_priority)

        优先级规则:
            - addon_priority: 从环境变量配置中获取，数值越小优先级越高
            - mock_priority: Mock模式下，Mock Bean=0, 非Mock Bean=1；非Mock模式下都为0
            - match_priority: 直接匹配=0, 实现类匹配=1
            - primary_priority: Primary Bean=0, 非Primary Bean=1
            - scope_priority: Factory Bean=0, 非Factory Bean=1
        """
        # 1. Addon优先级（数值越小优先级越高）
        addon_priority = AddonBeanOrderStrategy.get_addon_priority(bean_def)

        # 2. Mock优先级（仅在Mock模式下区分）
        if mock_mode:
            mock_priority = 0 if bean_def.is_mock else 1
        else:
            mock_priority = 0  # 非Mock模式下不区分

        # 3. 匹配方式优先级（直接匹配优先）
        match_priority = 0 if is_direct_match else 1

        # 4. Primary优先级（Primary优先）
        primary_priority = 0 if bean_def.is_primary else 1

        # 5. Scope优先级（Factory优先）
        scope_priority = 0 if bean_def.scope == BeanScope.FACTORY else 1

        return (
            addon_priority,
            mock_priority,
            match_priority,
            primary_priority,
            scope_priority,
        )

    @staticmethod
    def sort_beans_with_context(
        bean_defs: List[BeanDefinition],
        direct_match_types: Set[Type],
        mock_mode: bool = False,
    ) -> List[BeanDefinition]:
        """
        根据上下文信息对Bean定义列表进行排序（扩展版本）

        Args:
            bean_defs: Bean定义列表
            direct_match_types: 直接匹配的类型集合
            mock_mode: 是否处于Mock模式

        Returns:
            List[BeanDefinition]: 排序后的Bean定义列表

        注意:
            - 在非Mock模式下，Mock Bean会被直接过滤掉，不参与排序
            - 在Mock模式下，Mock Bean优先于非Mock Bean
            - addon_tag优先级最高，根据环境变量配置排序
        """
        # 在非Mock模式下，过滤掉所有Mock Bean
        if not mock_mode:
            bean_defs = [bd for bd in bean_defs if not bd.is_mock]

        # 为每个Bean计算排序键，然后按键排序
        sorted_beans = sorted(
            bean_defs,
            key=lambda bd: AddonBeanOrderStrategy.calculate_order_key(
                bean_def=bd,
                is_direct_match=bd.bean_type in direct_match_types,
                mock_mode=mock_mode,
            ),
        )
        return sorted_beans


# 模块加载时自动替换Bean排序策略
# 注意：这是一个临时方案，因为DI机制还没有完全建立
# 一旦引用addon机制就会自动启用AddonBeanOrderStrategy
def _replace_strategy():
    """自动替换Bean排序策略"""
    try:
        from core.di.container import DIContainer

        DIContainer.replace_bean_order_strategy(AddonBeanOrderStrategy)
        logger.warning(
            "⚠️ Bean排序策略已自动替换为 AddonBeanOrderStrategy，支持 addon_tag 优先级"
        )
        logger.info(
            "  📌 Addon优先级配置: %s (环境变量: ADDON_PRIORITY)",
            AddonBeanOrderStrategy._load_addon_priority_map(),
        )
    except Exception as e:
        logger.error("替换Bean排序策略失败: %s", e)


# 执行自动替换
_replace_strategy()
