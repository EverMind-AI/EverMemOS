# -*- coding: utf-8 -*-
"""
cd /Users/admin/memsys_opensource
PYTHONPATH=/Users/admin/memsys_opensource/src python -m pytest src/core/di/tests/test_bean_order_strategy.py -v

BeanOrderStrategy 测试模块

全面测试Bean排序策略的各种场景，包括：
- calculate_order_key 方法的各种优先级计算
- sort_beans_with_context 方法的综合排序和过滤逻辑
- sort_beans 方法的简单排序逻辑
"""

import pytest
from typing import Set, Type
from core.di.bean_definition import BeanDefinition, BeanScope
from core.di.bean_order_strategy import BeanOrderStrategy


# ==================== 测试辅助类 ====================


class ServiceA:
    """测试服务A"""

    pass


class ServiceB:
    """测试服务B"""

    pass


class ServiceC:
    """测试服务C"""

    pass


class ServiceD:
    """测试服务D"""

    pass


# ==================== calculate_order_key 方法测试 ====================


class TestCalculateOrderKey:
    """测试 calculate_order_key 方法"""

    def test_mock_priority_in_mock_mode(self):
        """测试Mock模式下，Mock Bean优先级高于非Mock Bean"""
        # 创建Mock Bean和非Mock Bean
        mock_bean = BeanDefinition(ServiceA, is_mock=True)
        normal_bean = BeanDefinition(ServiceA, is_mock=False)

        # 计算排序键（Mock模式，直接匹配）
        mock_key = BeanOrderStrategy.calculate_order_key(
            mock_bean, is_direct_match=True, mock_mode=True
        )
        normal_key = BeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=True
        )

        # 验证：Mock Bean的mock_priority=0，非Mock Bean的mock_priority=1
        assert mock_key[0] == 0  # Mock Bean的mock优先级
        assert normal_key[0] == 1  # 非Mock Bean的mock优先级
        assert mock_key < normal_key  # Mock Bean优先级更高

    def test_mock_priority_in_normal_mode(self):
        """测试非Mock模式下，Mock Bean和非Mock Bean的mock_priority都为0"""
        # 创建Mock Bean和非Mock Bean
        mock_bean = BeanDefinition(ServiceA, is_mock=True)
        normal_bean = BeanDefinition(ServiceA, is_mock=False)

        # 计算排序键（非Mock模式，直接匹配）
        mock_key = BeanOrderStrategy.calculate_order_key(
            mock_bean, is_direct_match=True, mock_mode=False
        )
        normal_key = BeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=False
        )

        # 验证：非Mock模式下，两者的mock_priority都为0
        assert mock_key[0] == 0
        assert normal_key[0] == 0

    def test_direct_match_priority(self):
        """测试直接匹配优先于实现类匹配"""
        bean = BeanDefinition(ServiceA)

        # 计算排序键
        direct_match_key = BeanOrderStrategy.calculate_order_key(
            bean, is_direct_match=True, mock_mode=False
        )
        impl_match_key = BeanOrderStrategy.calculate_order_key(
            bean, is_direct_match=False, mock_mode=False
        )

        # 验证：直接匹配的match_priority=0，实现类匹配的match_priority=1
        assert direct_match_key[1] == 0  # 直接匹配
        assert impl_match_key[1] == 1  # 实现类匹配
        assert direct_match_key < impl_match_key  # 直接匹配优先级更高

    def test_primary_priority(self):
        """测试Primary Bean优先于非Primary Bean"""
        primary_bean = BeanDefinition(ServiceA, is_primary=True)
        normal_bean = BeanDefinition(ServiceA, is_primary=False)

        # 计算排序键（直接匹配，非Mock模式）
        primary_key = BeanOrderStrategy.calculate_order_key(
            primary_bean, is_direct_match=True, mock_mode=False
        )
        normal_key = BeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=False
        )

        # 验证：Primary Bean的primary_priority=0，非Primary Bean的primary_priority=1
        assert primary_key[2] == 0  # Primary Bean
        assert normal_key[2] == 1  # 非Primary Bean
        assert primary_key < normal_key  # Primary Bean优先级更高

    def test_factory_scope_priority(self):
        """测试Factory Bean优先于非Factory Bean"""
        factory_bean = BeanDefinition(ServiceA, scope=BeanScope.FACTORY)
        singleton_bean = BeanDefinition(ServiceA, scope=BeanScope.SINGLETON)
        prototype_bean = BeanDefinition(ServiceA, scope=BeanScope.PROTOTYPE)

        # 计算排序键（直接匹配，非Mock模式）
        factory_key = BeanOrderStrategy.calculate_order_key(
            factory_bean, is_direct_match=True, mock_mode=False
        )
        singleton_key = BeanOrderStrategy.calculate_order_key(
            singleton_bean, is_direct_match=True, mock_mode=False
        )
        prototype_key = BeanOrderStrategy.calculate_order_key(
            prototype_bean, is_direct_match=True, mock_mode=False
        )

        # 验证：Factory Bean的scope_priority=0，其他的scope_priority=1
        assert factory_key[3] == 0  # Factory Bean
        assert singleton_key[3] == 1  # Singleton Bean
        assert prototype_key[3] == 1  # Prototype Bean
        assert factory_key < singleton_key  # Factory Bean优先级更高
        assert factory_key < prototype_key  # Factory Bean优先级更高

    def test_comprehensive_priority_ordering(self):
        """测试综合优先级排序：mock > match > primary > scope"""
        # 在Mock模式下，创建各种组合的Bean
        # 最高优先级：Mock + 直接匹配 + Primary + Factory
        bean1 = BeanDefinition(
            ServiceA, is_mock=True, is_primary=True, scope=BeanScope.FACTORY
        )
        key1 = BeanOrderStrategy.calculate_order_key(
            bean1, is_direct_match=True, mock_mode=True
        )

        # 次高优先级：Mock + 直接匹配 + Primary + 非Factory
        bean2 = BeanDefinition(
            ServiceA, is_mock=True, is_primary=True, scope=BeanScope.SINGLETON
        )
        key2 = BeanOrderStrategy.calculate_order_key(
            bean2, is_direct_match=True, mock_mode=True
        )

        # 第三优先级：Mock + 直接匹配 + 非Primary + Factory
        bean3 = BeanDefinition(
            ServiceA, is_mock=True, is_primary=False, scope=BeanScope.FACTORY
        )
        key3 = BeanOrderStrategy.calculate_order_key(
            bean3, is_direct_match=True, mock_mode=True
        )

        # 第四优先级：Mock + 实现类匹配 + Primary + Factory
        bean4 = BeanDefinition(
            ServiceA, is_mock=True, is_primary=True, scope=BeanScope.FACTORY
        )
        key4 = BeanOrderStrategy.calculate_order_key(
            bean4, is_direct_match=False, mock_mode=True
        )

        # 第五优先级：非Mock + 直接匹配 + Primary + Factory
        bean5 = BeanDefinition(
            ServiceA, is_mock=False, is_primary=True, scope=BeanScope.FACTORY
        )
        key5 = BeanOrderStrategy.calculate_order_key(
            bean5, is_direct_match=True, mock_mode=True
        )

        # 验证排序顺序 - key1 是最高优先级
        assert key1 < key2  # Factory优先于非Factory（同为Mock+直接+Primary）
        assert key1 < key3  # Primary优先于非Primary（同为Mock+直接+Factory）
        assert key1 < key4  # 直接匹配优先于实现类匹配（同为Mock+Primary+Factory）
        assert key1 < key5  # Mock优先于非Mock（同为直接+Primary+Factory）

        # 验证次级优先级 - 在优先级元组中，前面的位权重更高
        assert key2 < key3  # (0,0,0,1) < (0,0,1,0): scope差异 vs primary差异
        assert key3 < key4  # (0,0,1,0) < (0,1,0,0): primary差异 vs match差异
        assert key4 < key5  # (0,1,0,0) < (1,0,0,0): match差异 vs mock差异


# ==================== sort_beans_with_context 方法测试 ====================


class TestSortBeansWithContext:
    """测试 sort_beans_with_context 方法"""

    def test_filter_mock_beans_in_normal_mode(self):
        """测试非Mock模式下，Mock Bean被过滤掉"""
        # 创建包含Mock Bean和非Mock Bean的列表
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="mock_a", is_mock=True),
            BeanDefinition(ServiceB, bean_name="normal_b", is_mock=False),
            BeanDefinition(ServiceC, bean_name="mock_c", is_mock=True),
            BeanDefinition(ServiceD, bean_name="normal_d", is_mock=False),
        ]

        # 排序（非Mock模式）
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=set(), mock_mode=False
        )

        # 验证：只剩下非Mock Bean
        assert len(sorted_beans) == 2
        assert all(not bd.is_mock for bd in sorted_beans)
        assert {bd.bean_name for bd in sorted_beans} == {"normal_b", "normal_d"}

    def test_keep_mock_beans_in_mock_mode(self):
        """测试Mock模式下，Mock Bean被保留且优先"""
        # 创建包含Mock Bean和非Mock Bean的列表
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="normal_a", is_mock=False),
            BeanDefinition(ServiceB, bean_name="mock_b", is_mock=True),
            BeanDefinition(ServiceC, bean_name="mock_c", is_mock=True),
            BeanDefinition(ServiceD, bean_name="normal_d", is_mock=False),
        ]

        # 排序（Mock模式，所有都是直接匹配）
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=True,
        )

        # 验证：所有Bean都被保留
        assert len(sorted_beans) == 4

        # 验证：Mock Bean在前，非Mock Bean在后
        assert sorted_beans[0].is_mock
        assert sorted_beans[1].is_mock
        assert not sorted_beans[2].is_mock
        assert not sorted_beans[3].is_mock

    def test_direct_match_types_sorting(self):
        """测试直接匹配类型的优先排序"""
        # 创建Bean列表
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="impl_a"),
            BeanDefinition(ServiceB, bean_name="direct_b"),
            BeanDefinition(ServiceC, bean_name="impl_c"),
            BeanDefinition(ServiceD, bean_name="direct_d"),
        ]

        # 设置ServiceB和ServiceD为直接匹配类型
        direct_match_types = {ServiceB, ServiceD}

        # 排序（非Mock模式）
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=direct_match_types, mock_mode=False
        )

        # 验证：直接匹配的Bean在前
        assert sorted_beans[0].bean_type in direct_match_types
        assert sorted_beans[1].bean_type in direct_match_types
        assert sorted_beans[2].bean_type not in direct_match_types
        assert sorted_beans[3].bean_type not in direct_match_types

    def test_primary_beans_sorting(self):
        """测试Primary Bean的优先排序"""
        # 创建Bean列表（所有都是直接匹配，非Mock模式）
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="normal_a", is_primary=False),
            BeanDefinition(ServiceB, bean_name="primary_b", is_primary=True),
            BeanDefinition(ServiceC, bean_name="normal_c", is_primary=False),
            BeanDefinition(ServiceD, bean_name="primary_d", is_primary=True),
        ]

        # 排序（所有都是直接匹配）
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # 验证：Primary Bean在前
        assert sorted_beans[0].is_primary
        assert sorted_beans[1].is_primary
        assert not sorted_beans[2].is_primary
        assert not sorted_beans[3].is_primary

    def test_factory_scope_sorting(self):
        """测试Factory Bean的优先排序"""
        # 创建Bean列表（所有都是直接匹配，非Primary，非Mock）
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="singleton_a", scope=BeanScope.SINGLETON
            ),
            BeanDefinition(ServiceB, bean_name="factory_b", scope=BeanScope.FACTORY),
            BeanDefinition(
                ServiceC, bean_name="prototype_c", scope=BeanScope.PROTOTYPE
            ),
            BeanDefinition(ServiceD, bean_name="factory_d", scope=BeanScope.FACTORY),
        ]

        # 排序（所有都是直接匹配）
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # 验证：Factory Bean在前
        assert sorted_beans[0].scope == BeanScope.FACTORY
        assert sorted_beans[1].scope == BeanScope.FACTORY
        assert sorted_beans[2].scope in {BeanScope.SINGLETON, BeanScope.PROTOTYPE}
        assert sorted_beans[3].scope in {BeanScope.SINGLETON, BeanScope.PROTOTYPE}

    def test_comprehensive_sorting(self):
        """测试综合排序场景：mock + match + primary + scope"""
        # 创建各种组合的Bean列表
        bean_defs = [
            # 最低优先级：非直接匹配 + 非Primary + 非Factory
            BeanDefinition(
                ServiceA,
                bean_name="lowest",
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # 中等优先级：直接匹配 + 非Primary + 非Factory
            BeanDefinition(
                ServiceB,
                bean_name="medium1",
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # 较高优先级：直接匹配 + Primary + 非Factory
            BeanDefinition(
                ServiceC, bean_name="high1", is_primary=True, scope=BeanScope.SINGLETON
            ),
            # 最高优先级：直接匹配 + Primary + Factory
            BeanDefinition(
                ServiceD, bean_name="highest", is_primary=True, scope=BeanScope.FACTORY
            ),
            # 次高优先级：直接匹配 + 非Primary + Factory
            BeanDefinition(
                ServiceA, bean_name="high2", is_primary=False, scope=BeanScope.FACTORY
            ),
        ]

        # 设置直接匹配类型（ServiceB, ServiceC, ServiceD，但不包括ServiceA）
        direct_match_types = {ServiceB, ServiceC, ServiceD}

        # 排序（非Mock模式）
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=direct_match_types, mock_mode=False
        )

        # 验证排序顺序
        assert sorted_beans[0].bean_name == "highest"  # 直接+Primary+Factory
        assert sorted_beans[1].bean_name in {
            "high1",
            "high2",
        }  # 直接+Primary+非Factory 或 直接+非Primary+Factory
        assert sorted_beans[4].bean_name == "lowest"  # 非直接+非Primary+非Factory

    def test_empty_list(self):
        """测试空列表"""
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=[], direct_match_types=set(), mock_mode=False
        )
        assert sorted_beans == []

    def test_single_bean(self):
        """测试单个Bean"""
        bean_defs = [BeanDefinition(ServiceA, bean_name="single")]
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )
        assert len(sorted_beans) == 1
        assert sorted_beans[0].bean_name == "single"

    def test_all_mock_beans_in_normal_mode(self):
        """测试非Mock模式下，所有Bean都是Mock Bean的情况"""
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="mock_a", is_mock=True),
            BeanDefinition(ServiceB, bean_name="mock_b", is_mock=True),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=set(), mock_mode=False
        )

        # 验证：所有Mock Bean被过滤，结果为空
        assert sorted_beans == []

    def test_complex_mock_mode_sorting(self):
        """测试Mock模式下的复杂排序"""
        bean_defs = [
            # 非Mock + 非直接 + 非Primary + 非Factory（最低）
            BeanDefinition(
                ServiceA,
                bean_name="lowest",
                is_mock=False,
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # Mock + 非直接 + 非Primary + 非Factory（中等）
            BeanDefinition(
                ServiceB,
                bean_name="medium",
                is_mock=True,
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # Mock + 直接 + 非Primary + 非Factory（较高）
            BeanDefinition(
                ServiceC,
                bean_name="high",
                is_mock=True,
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # Mock + 直接 + Primary + Factory（最高）
            BeanDefinition(
                ServiceD,
                bean_name="highest",
                is_mock=True,
                is_primary=True,
                scope=BeanScope.FACTORY,
            ),
        ]

        # 排序（Mock模式，ServiceC和ServiceD为直接匹配）
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceC, ServiceD}, mock_mode=True
        )

        # 验证排序顺序
        assert sorted_beans[0].bean_name == "highest"  # Mock+直接+Primary+Factory
        assert sorted_beans[1].bean_name == "high"  # Mock+直接+非Primary+非Factory
        assert sorted_beans[2].bean_name == "medium"  # Mock+非直接+非Primary+非Factory
        assert (
            sorted_beans[3].bean_name == "lowest"
        )  # 非Mock+非直接+非Primary+非Factory


# ==================== sort_beans 方法测试 ====================


class TestSortBeans:
    """测试 sort_beans 方法（简单排序）"""

    def test_primary_priority_simple(self):
        """测试Primary Bean优先于非Primary Bean（简单排序）"""
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="normal_a", is_primary=False),
            BeanDefinition(ServiceB, bean_name="primary_b", is_primary=True),
            BeanDefinition(ServiceC, bean_name="normal_c", is_primary=False),
            BeanDefinition(ServiceD, bean_name="primary_d", is_primary=True),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # 验证：Primary Bean在前
        assert sorted_beans[0].is_primary
        assert sorted_beans[1].is_primary
        assert not sorted_beans[2].is_primary
        assert not sorted_beans[3].is_primary

    def test_factory_scope_priority_simple(self):
        """测试Factory Bean优先于非Factory Bean（简单排序）"""
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="singleton_a", scope=BeanScope.SINGLETON
            ),
            BeanDefinition(ServiceB, bean_name="factory_b", scope=BeanScope.FACTORY),
            BeanDefinition(
                ServiceC, bean_name="prototype_c", scope=BeanScope.PROTOTYPE
            ),
            BeanDefinition(ServiceD, bean_name="factory_d", scope=BeanScope.FACTORY),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # 验证：Factory Bean在前
        assert sorted_beans[0].scope == BeanScope.FACTORY
        assert sorted_beans[1].scope == BeanScope.FACTORY
        assert sorted_beans[2].scope in {BeanScope.SINGLETON, BeanScope.PROTOTYPE}
        assert sorted_beans[3].scope in {BeanScope.SINGLETON, BeanScope.PROTOTYPE}

    def test_primary_and_factory_combined(self):
        """测试Primary和Factory优先级组合（简单排序）"""
        bean_defs = [
            # 最低优先级：非Primary + 非Factory
            BeanDefinition(
                ServiceA,
                bean_name="lowest",
                is_primary=False,
                scope=BeanScope.SINGLETON,
            ),
            # 中等优先级：非Primary + Factory
            BeanDefinition(
                ServiceB, bean_name="medium", is_primary=False, scope=BeanScope.FACTORY
            ),
            # 较高优先级：Primary + 非Factory
            BeanDefinition(
                ServiceC, bean_name="high", is_primary=True, scope=BeanScope.SINGLETON
            ),
            # 最高优先级：Primary + Factory
            BeanDefinition(
                ServiceD, bean_name="highest", is_primary=True, scope=BeanScope.FACTORY
            ),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # 验证排序顺序
        assert sorted_beans[0].bean_name == "highest"  # Primary+Factory
        assert sorted_beans[1].bean_name == "high"  # Primary+非Factory
        assert sorted_beans[2].bean_name == "medium"  # 非Primary+Factory
        assert sorted_beans[3].bean_name == "lowest"  # 非Primary+非Factory

    def test_mock_beans_not_filtered_in_simple_sort(self):
        """测试简单排序不过滤Mock Bean"""
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="mock_a", is_mock=True, is_primary=False
            ),
            BeanDefinition(
                ServiceB, bean_name="normal_b", is_mock=False, is_primary=True
            ),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # 验证：Mock Bean不被过滤（简单排序不考虑Mock）
        assert len(sorted_beans) == 2
        # 验证：Primary优先（不管是否Mock）
        assert sorted_beans[0].bean_name == "normal_b"
        assert sorted_beans[1].bean_name == "mock_a"

    def test_empty_list_simple(self):
        """测试空列表（简单排序）"""
        sorted_beans = BeanOrderStrategy.sort_beans([])
        assert sorted_beans == []

    def test_single_bean_simple(self):
        """测试单个Bean（简单排序）"""
        bean_defs = [BeanDefinition(ServiceA, bean_name="single")]
        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)
        assert len(sorted_beans) == 1
        assert sorted_beans[0].bean_name == "single"

    def test_same_priority_beans(self):
        """测试相同优先级的Bean保持原有顺序"""
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="a", is_primary=False, scope=BeanScope.SINGLETON
            ),
            BeanDefinition(
                ServiceB, bean_name="b", is_primary=False, scope=BeanScope.SINGLETON
            ),
            BeanDefinition(
                ServiceC, bean_name="c", is_primary=False, scope=BeanScope.SINGLETON
            ),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans(bean_defs)

        # 验证：相同优先级的Bean保持原有顺序（stable sort）
        assert sorted_beans[0].bean_name == "a"
        assert sorted_beans[1].bean_name == "b"
        assert sorted_beans[2].bean_name == "c"


# ==================== 边界情况测试 ====================


class TestEdgeCases:
    """测试边界情况和异常场景"""

    def test_none_direct_match_types(self):
        """测试direct_match_types为空集合"""
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="a"),
            BeanDefinition(ServiceB, bean_name="b"),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=set(), mock_mode=False
        )

        # 验证：所有Bean都被视为非直接匹配
        assert len(sorted_beans) == 2

    def test_all_direct_match_types(self):
        """测试所有Bean都是直接匹配"""
        bean_defs = [
            BeanDefinition(ServiceA, bean_name="a", is_primary=True),
            BeanDefinition(ServiceB, bean_name="b", is_primary=False),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB},
            mock_mode=False,
        )

        # 验证：Primary Bean优先
        assert sorted_beans[0].bean_name == "a"
        assert sorted_beans[1].bean_name == "b"

    def test_multiple_beans_same_type(self):
        """测试同一类型的多个Bean"""
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="a1", is_primary=False, scope=BeanScope.SINGLETON
            ),
            BeanDefinition(
                ServiceA, bean_name="a2", is_primary=True, scope=BeanScope.SINGLETON
            ),
            BeanDefinition(
                ServiceA, bean_name="a3", is_primary=False, scope=BeanScope.FACTORY
            ),
            BeanDefinition(
                ServiceA, bean_name="a4", is_primary=True, scope=BeanScope.FACTORY
            ),
        ]

        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )

        # 验证排序顺序：Primary+Factory > Primary+非Factory > 非Primary+Factory > 非Primary+非Factory
        assert sorted_beans[0].bean_name == "a4"  # Primary+Factory
        assert sorted_beans[1].bean_name == "a2"  # Primary+非Factory
        assert sorted_beans[2].bean_name == "a3"  # 非Primary+Factory
        assert sorted_beans[3].bean_name == "a1"  # 非Primary+非Factory

    def test_all_attributes_combinations(self):
        """测试所有属性的组合（2^4=16种组合）"""
        # 生成所有可能的组合
        combinations = []
        for i in range(16):
            is_mock = bool(i & 8)
            is_direct = bool(i & 4)
            is_primary = bool(i & 2)
            is_factory = bool(i & 1)

            bean_type = [ServiceA, ServiceB, ServiceC, ServiceD][i % 4]
            bean = BeanDefinition(
                bean_type,
                bean_name=f"bean_{i}",
                is_mock=is_mock,
                is_primary=is_primary,
                scope=BeanScope.FACTORY if is_factory else BeanScope.SINGLETON,
            )
            combinations.append((bean, is_direct, bean_type))

        # 提取Bean列表和直接匹配类型
        bean_defs = [c[0] for c in combinations]
        direct_match_types = {c[2] for c in combinations if c[1]}

        # 排序（Mock模式）
        sorted_beans = BeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types=direct_match_types, mock_mode=True
        )

        # 验证：按优先级排序，Mock Bean在前
        # 验证所有Bean都被保留
        assert len(sorted_beans) == 16

        # 验证第一个Bean应该是优先级最高的
        first_bean = sorted_beans[0]
        first_key = BeanOrderStrategy.calculate_order_key(
            first_bean,
            is_direct_match=first_bean.bean_type in direct_match_types,
            mock_mode=True,
        )

        # 验证所有其他Bean的优先级都不高于第一个Bean
        for bean in sorted_beans[1:]:
            bean_key = BeanOrderStrategy.calculate_order_key(
                bean,
                is_direct_match=bean.bean_type in direct_match_types,
                mock_mode=True,
            )
            assert first_key <= bean_key


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
