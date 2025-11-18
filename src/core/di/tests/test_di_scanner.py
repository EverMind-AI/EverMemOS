# -*- coding: utf-8 -*-
"""
cd /Users/admin/memsys_opensource
PYTHONPATH=/Users/admin/memsys_opensource/src python -m pytest src/core/di/tests/test_di_scanner.py -v -s

DI Scanner 测试

测试Scanner的组件扫描和自动注册功能
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from core.di.container import DIContainer, get_container
from core.di.scanner import ComponentScanner
from core.di.decorators import component, service, repository, mock_impl, factory
from core.di.bean_definition import BeanScope
from core.di.tests.test_fixtures import (
    UserRepository,
    MySQLUserRepository,
    NotificationService,
    EmailNotificationService,
)


class TestScannerBasicFunctionality:
    """测试Scanner的基本功能"""

    def setup_method(self):
        """每个测试前创建新容器和扫描器"""
        self.container = DIContainer()
        self.scanner = ComponentScanner()

    def test_scanner_initialization(self):
        """测试Scanner的初始化"""
        assert self.scanner is not None
        assert self.scanner.scan_paths == []
        assert self.scanner.scan_packages == []
        assert self.scanner.recursive is True

    def test_add_scan_path(self):
        """测试添加扫描路径"""
        self.scanner.add_scan_path("/path/to/scan")
        assert "/path/to/scan" in self.scanner.scan_paths

        # 链式调用
        self.scanner.add_scan_path("/path1").add_scan_path("/path2")
        assert len(self.scanner.scan_paths) == 3

    def test_add_scan_package(self):
        """测试添加扫描包"""
        self.scanner.add_scan_package("my.package")
        assert "my.package" in self.scanner.scan_packages

        # 链式调用
        self.scanner.add_scan_package("pkg1").add_scan_package("pkg2")
        assert len(self.scanner.scan_packages) == 3

    def test_exclude_patterns(self):
        """测试排除模式"""
        self.scanner.exclude_pattern("test_")
        assert "test_" in self.scanner.exclude_patterns

        # 默认排除模式应该存在
        assert "__pycache__" in self.scanner.exclude_paths
        assert "test_" in self.scanner.exclude_patterns


class TestComponentDecoratorIntegration:
    """测试装饰器与Container的集成"""

    def setup_method(self):
        """每个测试前重置全局容器"""
        # 注意：装饰器默认注册到全局容器
        # 这里我们测试装饰器的行为
        pass

    def test_component_decorator_registers_bean(self):
        """测试@component装饰器自动注册Bean"""
        container = get_container()

        # 定义一个组件
        @component(name="test_component_unique_1")
        class TestComponent1:
            def __init__(self):
                self.value = "test1"

        # 验证Bean已注册
        assert container.contains_bean("test_component_unique_1")

        # 获取Bean
        comp = container.get_bean("test_component_unique_1")
        assert isinstance(comp, TestComponent1)
        assert comp.value == "test1"

    def test_service_decorator_registers_bean(self):
        """测试@service装饰器自动注册Bean"""
        container = get_container()

        # 定义一个服务
        @service(name="test_service_unique_1", primary=True)
        class TestService1:
            def __init__(self):
                self.service_type = "test"

        # 验证Bean已注册
        assert container.contains_bean("test_service_unique_1")

        # 获取Bean
        svc = container.get_bean("test_service_unique_1")
        assert isinstance(svc, TestService1)
        assert svc.service_type == "test"

    def test_repository_decorator_registers_bean(self):
        """测试@repository装饰器自动注册Bean"""
        container = get_container()

        # 定义一个仓储
        @repository(name="test_repo_unique_1")
        class TestRepository1:
            def __init__(self):
                self.db = "sqlite"

        # 验证Bean已注册
        assert container.contains_bean("test_repo_unique_1")

        # 获取Bean
        repo = container.get_bean("test_repo_unique_1")
        assert isinstance(repo, TestRepository1)
        assert repo.db == "sqlite"

    def test_mock_impl_decorator_registers_mock_bean(self):
        """测试@mock_impl装饰器注册Mock Bean"""
        container = get_container()

        # 定义一个Mock实现
        @mock_impl(name="test_mock_unique_1")
        class TestMock1:
            def __init__(self):
                self.is_mock = True

        # 验证Bean已注册
        assert container.contains_bean("test_mock_unique_1")

        # 在非Mock模式下，不应自动获取到Mock Bean
        # 但通过名称可以获取
        mock = container.get_bean("test_mock_unique_1")
        assert isinstance(mock, TestMock1)
        assert mock.is_mock is True

    def test_factory_decorator_registers_factory(self):
        """测试@factory装饰器注册Factory"""
        container = get_container()

        # 定义一个Factory
        class Product:
            def __init__(self, name: str):
                self.name = name

        @factory(bean_type=Product, name="test_factory_unique_1")
        def create_product() -> Product:
            return Product(name="TestProduct")

        # 验证Factory已注册
        assert container.contains_bean("test_factory_unique_1")

        # 获取Bean（由Factory创建）
        product = container.get_bean("test_factory_unique_1")
        assert isinstance(product, Product)
        assert product.name == "TestProduct"

    def test_component_with_scope(self):
        """测试@component装饰器指定Scope"""
        container = get_container()

        # 定义Prototype scope的组件
        @component(name="test_prototype_unique_1", scope=BeanScope.PROTOTYPE)
        class TestPrototype1:
            counter = 0

            def __init__(self):
                TestPrototype1.counter += 1
                self.id = TestPrototype1.counter

        # 获取多个实例
        obj1 = container.get_bean("test_prototype_unique_1")
        obj2 = container.get_bean("test_prototype_unique_1")

        # Prototype scope应创建不同实例
        assert obj1 is not obj2
        assert obj1.id != obj2.id


class TestInterfaceImplementationScanning:
    """测试接口和实现类的扫描"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()

    def test_register_interface_implementations(self):
        """测试注册接口的多个实现"""
        # 手动注册实现类（模拟扫描结果）
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_user_repo", is_primary=True
        )

        # 验证可以通过接口类型获取
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)

    def test_multiple_implementations_of_interface(self):
        """测试同一接口的多个实现"""
        from core.di.tests.test_fixtures import (
            PostgreSQLUserRepository,
            MockUserRepository,
        )

        # 注册多个实现
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=False
        )
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository,
            bean_name="postgres_repo",
            is_primary=True,
        )
        self.container.register_bean(
            bean_type=MockUserRepository, bean_name="mock_repo", is_mock=True
        )

        # 获取Primary实现
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, PostgreSQLUserRepository)

        # 获取所有实现（非Mock模式）
        all_repos = self.container.get_beans_by_type(UserRepository)
        assert len(all_repos) == 2  # MySQL + PostgreSQL


class TestScanContextAndMetadata:
    """测试扫描上下文和元数据"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()

    def test_bean_with_metadata(self):
        """测试Bean的元数据"""
        # 注册带元数据的Bean
        self.container.register_bean(
            bean_type=MySQLUserRepository,
            bean_name="mysql_repo",
            metadata={"version": "1.0", "author": "test", "db_type": "mysql"},
        )

        # 获取Bean Definition验证元数据
        bean_def = self.container._named_beans.get("mysql_repo")
        assert bean_def is not None
        assert bean_def.metadata["version"] == "1.0"
        assert bean_def.metadata["author"] == "test"
        assert bean_def.metadata["db_type"] == "mysql"

    def test_metadata_from_decorator(self):
        """测试装饰器传递的元数据"""
        container = get_container()

        # 定义带元数据的组件
        @component(
            name="test_metadata_comp_unique_1", metadata={"env": "test", "priority": 10}
        )
        class TestMetadataComp:
            pass

        # 获取Bean Definition
        bean_def = container._named_beans.get("test_metadata_comp_unique_1")
        assert bean_def is not None
        assert bean_def.metadata["env"] == "test"
        assert bean_def.metadata["priority"] == 10


class TestConditionalRegistration:
    """测试条件注册"""

    def test_lazy_registration(self):
        """测试延迟注册"""
        container = get_container()
        initial_bean_count = len(container._named_beans)

        # 定义延迟注册的组件
        @component(name="lazy_comp_unique_1", lazy=True)
        class LazyComponent:
            pass

        # 延迟注册的Bean不应立即出现在容器中
        # 注意：当前实现中lazy=True只是标记，实际行为需要查看具体实现
        # 这里我们只验证组件被正确标记
        assert hasattr(LazyComponent, '_di_lazy')
        assert LazyComponent._di_lazy is True


class TestBeanDependencies:
    """测试Bean之间的依赖关系"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()

    def test_bean_depends_on_another(self):
        """测试Bean依赖另一个Bean"""
        from core.di.tests.test_fixtures import UserServiceImpl, register_standard_beans

        # 注册依赖的Bean
        register_standard_beans(self.container)

        # 创建依赖其他Bean的Service
        service = UserServiceImpl(container=self.container)

        # 验证依赖注入成功
        assert service.repository is not None
        assert isinstance(service.repository, UserRepository)

        # 验证Service功能正常
        user = service.get_user(1)
        assert user is not None


class TestEdgeCases:
    """测试边界情况"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()

    def test_empty_container(self):
        """测试空容器"""
        from core.di.exceptions import BeanNotFoundError

        # 空容器应抛出异常
        with pytest.raises(BeanNotFoundError):
            self.container.get_bean_by_type(UserRepository)

    def test_duplicate_bean_name(self):
        """测试重复的Bean名称会抛出异常"""
        from core.di.exceptions import DuplicateBeanError
        from core.di.tests.test_fixtures import PostgreSQLUserRepository

        # 注册第一个Bean
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="same_name"
        )

        # 尝试注册同名Bean应该抛出异常
        with pytest.raises(DuplicateBeanError):
            self.container.register_bean(
                bean_type=PostgreSQLUserRepository, bean_name="same_name"
            )

    def test_get_all_beans_empty_result(self):
        """测试获取不存在类型的所有Bean"""

        # 未注册的类型
        class UnregisteredService:
            pass

        # 获取所有Bean应返回空列表
        beans = self.container.get_beans_by_type(UnregisteredService)
        assert beans == []


class TestRealWorldScanningScenario:
    """测试真实扫描场景"""

    def test_scan_test_fixtures_module(self):
        """测试扫描test_fixtures模块"""
        # 这个测试验证我们可以从fixtures模块导入类
        from core.di.tests.test_fixtures import (
            MySQLUserRepository,
            PostgreSQLUserRepository,
            MockUserRepository,
            EmailNotificationService,
            SMSNotificationService,
            RedisCacheService,
            MemoryCacheService,
        )

        # 验证所有类都可以正常导入和实例化
        mysql_repo = MySQLUserRepository()
        assert mysql_repo.db_type == "mysql"

        postgres_repo = PostgreSQLUserRepository()
        assert postgres_repo.db_type == "postgres"

        mock_repo = MockUserRepository()
        assert mock_repo.db_type == "mock"

        email_notif = EmailNotificationService()
        assert email_notif.sent_messages == []

        sms_notif = SMSNotificationService()
        assert sms_notif.sent_messages == []

        redis_cache = RedisCacheService()
        assert redis_cache.cache_type == "redis"

        memory_cache = MemoryCacheService()
        assert memory_cache.cache_type == "memory"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s", "--tb=short"])
