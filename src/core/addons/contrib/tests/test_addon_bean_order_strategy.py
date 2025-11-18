# -*- coding: utf-8 -*-
"""
cd /Users/admin/memsys_opensource
PYTHONPATH=/Users/admin/memsys_opensource/src python -m pytest src/core/addons/contrib/tests/test_addon_bean_order_strategy.py -v -s

Addon Bean排序策略测试

测试AddonBeanOrderStrategy的扩展功能，特别是addon_tag优先级
"""

import os
import pytest
from typing import Set, Type
from core.di.bean_definition import BeanDefinition, BeanScope
from core.addons.contrib.addon_bean_order_strategy import AddonBeanOrderStrategy


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


# ==================== 测试Addon优先级配置 ====================


class TestAddonPriorityConfiguration:
    """测试Addon优先级配置的加载"""

    def setup_method(self):
        """每个测试前重置优先级缓存"""
        AddonBeanOrderStrategy._addon_priority_map = None

    def teardown_method(self):
        """每个测试后恢复环境变量"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_load_default_priority_map(self):
        """测试加载默认的优先级配置"""
        # 不设置环境变量，应使用默认配置
        priority_map = AddonBeanOrderStrategy._load_addon_priority_map()

        # 验证默认配置：core:1000,enterprise:50
        assert "core" in priority_map
        assert "enterprise" in priority_map
        assert priority_map["core"] == 1000
        assert priority_map["enterprise"] == 50

    def test_load_custom_priority_map(self):
        """测试从环境变量加载自定义优先级配置"""
        # 设置环境变量
        os.environ["ADDON_PRIORITY"] = "addon1:100,addon2:200,addon3:50"

        # 重新加载
        AddonBeanOrderStrategy._addon_priority_map = None
        priority_map = AddonBeanOrderStrategy._load_addon_priority_map()

        # 验证配置
        assert priority_map["addon1"] == 100
        assert priority_map["addon2"] == 200
        assert priority_map["addon3"] == 50

    def test_priority_map_caching(self):
        """测试优先级配置的缓存机制"""
        # 第一次加载
        priority_map1 = AddonBeanOrderStrategy._load_addon_priority_map()

        # 第二次加载（应该从缓存返回）
        priority_map2 = AddonBeanOrderStrategy._load_addon_priority_map()

        # 应该是同一个对象
        assert priority_map1 is priority_map2

    def test_invalid_priority_config_ignored(self):
        """测试无效的优先级配置会被忽略"""
        # 设置包含无效值的环境变量
        os.environ["ADDON_PRIORITY"] = "valid:100,invalid:abc,another:200"

        # 重新加载
        AddonBeanOrderStrategy._addon_priority_map = None
        priority_map = AddonBeanOrderStrategy._load_addon_priority_map()

        # 验证：有效的配置被加载，无效的被忽略
        assert priority_map["valid"] == 100
        assert priority_map["another"] == 200
        assert "invalid" not in priority_map

    def test_priority_config_with_spaces(self):
        """测试配置中包含空格的情况"""
        # 设置包含空格的环境变量
        os.environ["ADDON_PRIORITY"] = " addon1 : 100 , addon2 : 200 "

        # 重新加载
        AddonBeanOrderStrategy._addon_priority_map = None
        priority_map = AddonBeanOrderStrategy._load_addon_priority_map()

        # 验证：空格被正确处理
        assert priority_map["addon1"] == 100
        assert priority_map["addon2"] == 200


# ==================== 测试获取Addon优先级 ====================


class TestGetAddonPriority:
    """测试获取Bean的Addon优先级"""

    def setup_method(self):
        """每个测试前重置配置"""
        AddonBeanOrderStrategy._addon_priority_map = None
        os.environ["ADDON_PRIORITY"] = "core:1000,enterprise:50,custom:200"

    def teardown_method(self):
        """每个测试后清理"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_get_priority_with_addon_tag(self):
        """测试获取有addon_tag的Bean优先级"""
        # 创建带addon_tag的Bean
        bean_core = BeanDefinition(ServiceA, metadata={"addon_tag": "core"})
        bean_enterprise = BeanDefinition(ServiceB, metadata={"addon_tag": "enterprise"})
        bean_custom = BeanDefinition(ServiceC, metadata={"addon_tag": "custom"})

        # 获取优先级
        priority_core = AddonBeanOrderStrategy.get_addon_priority(bean_core)
        priority_enterprise = AddonBeanOrderStrategy.get_addon_priority(bean_enterprise)
        priority_custom = AddonBeanOrderStrategy.get_addon_priority(bean_custom)

        # 验证
        assert priority_core == 1000
        assert priority_enterprise == 50
        assert priority_custom == 200

        # 验证优先级排序：enterprise < custom < core
        assert priority_enterprise < priority_custom < priority_core

    def test_get_priority_without_addon_tag(self):
        """测试获取没有addon_tag的Bean优先级"""
        # 创建没有addon_tag的Bean
        bean_no_tag = BeanDefinition(ServiceA)

        # 获取优先级
        priority = AddonBeanOrderStrategy.get_addon_priority(bean_no_tag)

        # 验证：返回最低优先级
        assert priority == 99999

    def test_get_priority_with_unknown_addon_tag(self):
        """测试获取未配置的addon_tag优先级"""
        # 创建带未配置的addon_tag的Bean
        bean_unknown = BeanDefinition(ServiceA, metadata={"addon_tag": "unknown_addon"})

        # 获取优先级
        priority = AddonBeanOrderStrategy.get_addon_priority(bean_unknown)

        # 验证：返回最低优先级
        assert priority == 99999

    def test_get_priority_with_empty_addon_tag(self):
        """测试获取空的addon_tag优先级"""
        # 创建带空addon_tag的Bean
        bean_empty = BeanDefinition(ServiceA, metadata={"addon_tag": ""})

        # 获取优先级
        priority = AddonBeanOrderStrategy.get_addon_priority(bean_empty)

        # 验证：返回最低优先级
        assert priority == 99999


# ==================== 测试计算排序键 ====================


class TestCalculateOrderKey:
    """测试计算扩展的排序键（包含addon优先级）"""

    def setup_method(self):
        """每个测试前重置配置"""
        AddonBeanOrderStrategy._addon_priority_map = None
        os.environ["ADDON_PRIORITY"] = "enterprise:50,core:1000"

    def teardown_method(self):
        """每个测试后清理"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_order_key_includes_addon_priority(self):
        """测试排序键包含addon优先级"""
        # 创建带addon_tag的Bean
        bean_enterprise = BeanDefinition(ServiceA, metadata={"addon_tag": "enterprise"})
        bean_core = BeanDefinition(ServiceB, metadata={"addon_tag": "core"})

        # 计算排序键
        key_enterprise = AddonBeanOrderStrategy.calculate_order_key(
            bean_enterprise, is_direct_match=True, mock_mode=False
        )
        key_core = AddonBeanOrderStrategy.calculate_order_key(
            bean_core, is_direct_match=True, mock_mode=False
        )

        # 验证：排序键为5元组（addon, mock, match, primary, scope）
        assert len(key_enterprise) == 5
        assert len(key_core) == 5

        # 验证：第一个元素是addon优先级
        assert key_enterprise[0] == 50  # enterprise
        assert key_core[0] == 1000  # core

        # 验证：enterprise优先级高于core
        assert key_enterprise < key_core

    def test_addon_priority_overrides_other_priorities(self):
        """测试addon优先级高于其他所有优先级"""
        # 创建两个Bean：
        # Bean1: enterprise addon + 非Primary + 非Factory
        # Bean2: core addon + Primary + Factory
        bean1 = BeanDefinition(
            ServiceA,
            is_primary=False,
            scope=BeanScope.SINGLETON,
            metadata={"addon_tag": "enterprise"},
        )
        bean2 = BeanDefinition(
            ServiceB,
            is_primary=True,
            scope=BeanScope.FACTORY,
            metadata={"addon_tag": "core"},
        )

        # 计算排序键
        key1 = AddonBeanOrderStrategy.calculate_order_key(
            bean1, is_direct_match=True, mock_mode=False
        )
        key2 = AddonBeanOrderStrategy.calculate_order_key(
            bean2, is_direct_match=True, mock_mode=False
        )

        # 验证：即使bean2有Primary+Factory，bean1因为addon优先级更高仍然排在前面
        assert key1 < key2

    def test_order_key_with_mock_mode(self):
        """测试Mock模式下的排序键"""
        # 创建Mock Bean和非Mock Bean（都是enterprise addon）
        mock_bean = BeanDefinition(
            ServiceA, is_mock=True, metadata={"addon_tag": "enterprise"}
        )
        normal_bean = BeanDefinition(
            ServiceB, is_mock=False, metadata={"addon_tag": "enterprise"}
        )

        # 计算排序键（Mock模式）
        mock_key = AddonBeanOrderStrategy.calculate_order_key(
            mock_bean, is_direct_match=True, mock_mode=True
        )
        normal_key = AddonBeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=True
        )

        # 验证：addon优先级相同，Mock优先
        assert mock_key[0] == normal_key[0]  # addon优先级相同
        assert mock_key[1] < normal_key[1]  # mock优先级不同
        assert mock_key < normal_key

    def test_order_key_backward_compatible(self):
        """测试向后兼容：没有addon_tag时仍按原逻辑排序"""
        # 创建两个Bean：都没有addon_tag，一个Primary一个非Primary
        primary_bean = BeanDefinition(ServiceA, is_primary=True)
        normal_bean = BeanDefinition(ServiceB, is_primary=False)

        # 计算排序键
        primary_key = AddonBeanOrderStrategy.calculate_order_key(
            primary_bean, is_direct_match=True, mock_mode=False
        )
        normal_key = AddonBeanOrderStrategy.calculate_order_key(
            normal_bean, is_direct_match=True, mock_mode=False
        )

        # 验证：addon优先级都是99999（最低）
        assert primary_key[0] == 99999
        assert normal_key[0] == 99999

        # 验证：Primary Bean优先级更高
        assert primary_key < normal_key


# ==================== 测试Bean列表排序 ====================


class TestSortBeansWithContext:
    """测试带上下文的Bean列表排序（包含addon优先级）"""

    def setup_method(self):
        """每个测试前重置配置"""
        AddonBeanOrderStrategy._addon_priority_map = None
        os.environ["ADDON_PRIORITY"] = "enterprise:50,core:1000,custom:500"

    def teardown_method(self):
        """每个测试后清理"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_sort_by_addon_priority(self):
        """测试按addon优先级排序"""
        # 创建不同addon的Bean
        bean_defs = [
            BeanDefinition(
                ServiceA, bean_name="core_bean", metadata={"addon_tag": "core"}
            ),
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_bean",
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceC, bean_name="custom_bean", metadata={"addon_tag": "custom"}
            ),
            BeanDefinition(ServiceD, bean_name="no_addon_bean"),
        ]

        # 排序
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # 验证排序顺序：enterprise(50) < custom(500) < core(1000) < no_addon(99999)
        assert sorted_beans[0].bean_name == "enterprise_bean"
        assert sorted_beans[1].bean_name == "custom_bean"
        assert sorted_beans[2].bean_name == "core_bean"
        assert sorted_beans[3].bean_name == "no_addon_bean"

    def test_addon_priority_with_primary_and_scope(self):
        """测试addon优先级与Primary和Scope的组合"""
        # 创建各种组合的Bean
        bean_defs = [
            # 最高优先级：enterprise + Primary + Factory
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_primary_factory",
                is_primary=True,
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "enterprise"},
            ),
            # 次高：enterprise + 非Primary + 非Factory
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_normal",
                is_primary=False,
                scope=BeanScope.SINGLETON,
                metadata={"addon_tag": "enterprise"},
            ),
            # 中等：core + Primary + Factory
            BeanDefinition(
                ServiceC,
                bean_name="core_primary_factory",
                is_primary=True,
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "core"},
            ),
            # 最低：无addon + Primary + Factory
            BeanDefinition(
                ServiceD,
                bean_name="no_addon_primary_factory",
                is_primary=True,
                scope=BeanScope.FACTORY,
            ),
        ]

        # 排序
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # 验证：addon优先级最高，即使其他属性不同
        assert sorted_beans[0].bean_name == "enterprise_primary_factory"
        assert sorted_beans[1].bean_name == "enterprise_normal"
        assert sorted_beans[2].bean_name == "core_primary_factory"
        assert sorted_beans[3].bean_name == "no_addon_primary_factory"

    def test_same_addon_priority_then_by_primary(self):
        """测试相同addon优先级时按Primary排序"""
        # 创建相同addon的Bean
        bean_defs = [
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_normal",
                is_primary=False,
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_primary",
                is_primary=True,
                metadata={"addon_tag": "enterprise"},
            ),
        ]

        # 排序
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB},
            mock_mode=False,
        )

        # 验证：相同addon优先级时，Primary优先
        assert sorted_beans[0].bean_name == "enterprise_primary"
        assert sorted_beans[1].bean_name == "enterprise_normal"

    def test_filter_mock_beans_in_normal_mode(self):
        """测试非Mock模式下过滤Mock Bean（addon版本）"""
        # 创建包含Mock Bean的列表
        bean_defs = [
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_mock",
                is_mock=True,
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_normal",
                is_mock=False,
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceC,
                bean_name="core_mock",
                is_mock=True,
                metadata={"addon_tag": "core"},
            ),
        ]

        # 排序（非Mock模式）
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC},
            mock_mode=False,
        )

        # 验证：只保留非Mock Bean
        assert len(sorted_beans) == 1
        assert sorted_beans[0].bean_name == "enterprise_normal"

    def test_mock_beans_priority_in_mock_mode(self):
        """测试Mock模式下Mock Bean优先（addon版本）"""
        # 创建混合Bean列表
        bean_defs = [
            BeanDefinition(
                ServiceA,
                bean_name="core_normal",
                is_mock=False,
                metadata={"addon_tag": "core"},
            ),
            BeanDefinition(
                ServiceB,
                bean_name="enterprise_mock",
                is_mock=True,
                metadata={"addon_tag": "enterprise"},
            ),
            BeanDefinition(
                ServiceC,
                bean_name="enterprise_normal",
                is_mock=False,
                metadata={"addon_tag": "enterprise"},
            ),
        ]

        # 排序（Mock模式）
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC},
            mock_mode=True,
        )

        # 验证：相同addon下Mock优先，但不同addon仍按addon优先级
        assert sorted_beans[0].bean_name == "enterprise_mock"  # enterprise + mock
        assert sorted_beans[1].bean_name == "enterprise_normal"  # enterprise + normal
        assert sorted_beans[2].bean_name == "core_normal"  # core + normal

    def test_empty_bean_list(self):
        """测试空Bean列表"""
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=[], direct_match_types=set(), mock_mode=False
        )
        assert sorted_beans == []

    def test_single_bean(self):
        """测试单个Bean"""
        bean_defs = [BeanDefinition(ServiceA, metadata={"addon_tag": "enterprise"})]
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )
        assert len(sorted_beans) == 1


# ==================== 综合场景测试 ====================


class TestComplexScenarios:
    """测试复杂的综合场景"""

    def setup_method(self):
        """每个测试前重置配置"""
        AddonBeanOrderStrategy._addon_priority_map = None
        os.environ["ADDON_PRIORITY"] = "enterprise:10,plugin1:50,plugin2:100,core:1000"

    def teardown_method(self):
        """每个测试后清理"""
        if "ADDON_PRIORITY" in os.environ:
            del os.environ["ADDON_PRIORITY"]
        AddonBeanOrderStrategy._addon_priority_map = None

    def test_multi_addon_multi_implementation(self):
        """测试多个addon的多个实现"""
        # 模拟真实场景：多个addon都提供了同一接口的实现
        bean_defs = [
            # enterprise提供的实现（Primary + Factory）
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_impl",
                is_primary=True,
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "enterprise"},
            ),
            # plugin1提供的实现（Primary）
            BeanDefinition(
                ServiceB,
                bean_name="plugin1_impl",
                is_primary=True,
                metadata={"addon_tag": "plugin1"},
            ),
            # plugin2提供的实现
            BeanDefinition(
                ServiceC, bean_name="plugin2_impl", metadata={"addon_tag": "plugin2"}
            ),
            # core提供的实现（Factory）
            BeanDefinition(
                ServiceD,
                bean_name="core_impl",
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "core"},
            ),
        ]

        # 排序
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs,
            direct_match_types={ServiceA, ServiceB, ServiceC, ServiceD},
            mock_mode=False,
        )

        # 验证：按addon优先级排序
        assert sorted_beans[0].bean_name == "enterprise_impl"  # 10
        assert sorted_beans[1].bean_name == "plugin1_impl"  # 50
        assert sorted_beans[2].bean_name == "plugin2_impl"  # 100
        assert sorted_beans[3].bean_name == "core_impl"  # 1000

    def test_addon_override_scenario(self):
        """测试addon覆盖场景：高优先级addon覆盖低优先级addon"""
        # 创建Bean：enterprise覆盖core的实现
        bean_defs = [
            BeanDefinition(
                ServiceA,
                bean_name="core_default",
                is_primary=True,
                scope=BeanScope.FACTORY,
                metadata={"addon_tag": "core"},
            ),
            BeanDefinition(
                ServiceA,
                bean_name="enterprise_override",
                is_primary=False,
                scope=BeanScope.SINGLETON,
                metadata={"addon_tag": "enterprise"},
            ),
        ]

        # 排序
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )

        # 验证：enterprise虽然不是Primary也不是Factory，但因为addon优先级高，排在前面
        assert sorted_beans[0].bean_name == "enterprise_override"
        assert sorted_beans[1].bean_name == "core_default"

    def test_all_attributes_combination(self):
        """测试所有属性的组合"""
        # 创建16个Bean，覆盖所有可能的组合
        bean_defs = []
        counter = 0

        for addon_tag in ["enterprise", "core", None]:
            for is_mock in [False, True]:
                for is_primary in [False, True]:
                    for is_factory in [False, True]:
                        metadata = {"addon_tag": addon_tag} if addon_tag else {}
                        bean = BeanDefinition(
                            ServiceA,
                            bean_name=f"bean_{counter}",
                            is_mock=is_mock,
                            is_primary=is_primary,
                            scope=(
                                BeanScope.FACTORY if is_factory else BeanScope.SINGLETON
                            ),
                            metadata=metadata,
                        )
                        bean_defs.append(bean)
                        counter += 1

        # 排序（非Mock模式）
        sorted_beans = AddonBeanOrderStrategy.sort_beans_with_context(
            bean_defs=bean_defs, direct_match_types={ServiceA}, mock_mode=False
        )

        # 验证：非Mock模式下，Mock Bean被过滤
        assert all(not bean.is_mock for bean in sorted_beans)

        # 验证：第一个Bean应该是enterprise addon
        assert sorted_beans[0].metadata.get("addon_tag") == "enterprise"

        # 验证：最后一个Bean应该是没有addon_tag的
        assert sorted_beans[-1].metadata.get("addon_tag") is None


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s", "--tb=short"])
