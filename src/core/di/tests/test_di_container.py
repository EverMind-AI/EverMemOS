# -*- coding: utf-8 -*-
"""
cd /Users/admin/memsys_opensource
PYTHONPATH=/Users/admin/memsys_opensource/src python -m pytest src/core/di/tests/test_di_container.py -v -s

DI Container 集成测试

测试Container的Bean注册、解析、优先级选择等核心功能
"""

import pytest
from abc import ABC, abstractmethod
from typing import List
from core.di.container import DIContainer
from core.di.bean_definition import BeanScope
from core.di.exceptions import BeanNotFoundError
from core.di.tests.test_fixtures import (
    # 用户服务相关
    UserRepository,
    MySQLUserRepository,
    PostgreSQLUserRepository,
    MockUserRepository,
    UserService,
    UserServiceImpl,
    # 通知服务相关
    NotificationService,
    EmailNotificationService,
    SMSNotificationService,
    PushNotificationService,
    # 邮件服务相关
    EmailService,
    SMTPEmailService,
    # 数据库连接相关
    DatabaseConnection,
    create_database_connection,
    create_readonly_connection,
    # Prototype服务
    PrototypeService,
    # 缓存服务相关
    CacheService,
    RedisCacheService,
    MemoryCacheService,
    # 工具函数
    register_standard_beans,
)


class TestBeanRegistrationAndRetrieval:
    """测试Bean的注册和获取"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()

    def test_register_and_get_single_bean(self):
        """测试注册和获取单个Bean"""
        # 注册Bean
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo"
        )

        # 通过类型获取
        repo = self.container.get_bean_by_type(MySQLUserRepository)
        assert isinstance(repo, MySQLUserRepository)
        assert repo.db_type == "mysql"

        # 通过名称获取
        repo2 = self.container.get_bean("mysql_repo")
        assert repo is repo2  # 单例模式，应该是同一个实例

    def test_register_multiple_implementations(self):
        """测试注册同一接口的多个实现"""
        # 注册多个UserRepository实现
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=True
        )
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository, bean_name="postgres_repo"
        )

        # 获取接口类型，应返回Primary实现
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)

        # 获取所有实现
        all_repos = self.container.get_beans_by_type(UserRepository)
        assert len(all_repos) == 2
        assert isinstance(all_repos[0], MySQLUserRepository)  # Primary在前
        assert isinstance(all_repos[1], PostgreSQLUserRepository)

    def test_bean_not_found_error(self):
        """测试获取不存在的Bean时抛出异常"""

        # 定义未注册的接口
        class UnregisteredService(ABC):
            pass

        # 通过类型获取应抛出异常
        with pytest.raises(BeanNotFoundError):
            self.container.get_bean_by_type(UnregisteredService)

        # 通过名称获取应抛出异常
        with pytest.raises(BeanNotFoundError):
            self.container.get_bean("non_existent_bean")

    def test_contains_bean_check(self):
        """测试Bean存在性检查"""
        # 注册Bean
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo"
        )

        # 检查Bean是否存在
        assert self.container.contains_bean_by_type(MySQLUserRepository)
        # 注意：接口查找需要先build继承关系缓存
        # 直接查接口可能返回False，因为注册的是实现类
        # 正确的方式是get_bean_by_type会自动查找实现类
        assert self.container.contains_bean("mysql_repo")

        # 检查不存在的Bean
        assert not self.container.contains_bean_by_type(PostgreSQLUserRepository)
        assert not self.container.contains_bean("non_existent")


class TestPrimaryBeanSelection:
    """测试Primary Bean的选择逻辑"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()

    def test_primary_bean_priority(self):
        """测试Primary Bean优先于非Primary Bean"""
        # 注册两个实现：一个Primary，一个非Primary
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository,
            bean_name="postgres_repo",
            is_primary=False,
        )
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=True
        )

        # 获取UserRepository，应返回Primary实现
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)
        assert repo.db_type == "mysql"

    def test_multiple_primary_beans_return_first(self):
        """测试多个Primary Bean时返回第一个注册的"""
        # 注册两个Primary实现
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=True
        )
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository,
            bean_name="postgres_repo",
            is_primary=True,
        )

        # 应返回第一个Primary Bean
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)

    def test_no_primary_bean_returns_first(self):
        """测试没有Primary Bean时返回第一个注册的"""
        # 注册两个非Primary实现
        self.container.register_bean(
            bean_type=MySQLUserRepository, bean_name="mysql_repo", is_primary=False
        )
        self.container.register_bean(
            bean_type=PostgreSQLUserRepository,
            bean_name="postgres_repo",
            is_primary=False,
        )

        # 应返回第一个注册的Bean
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MySQLUserRepository)


class TestMockMode:
    """测试Mock模式"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()
        # 注册标准Bean（包括Mock）
        register_standard_beans(self.container)

    def test_normal_mode_filters_mock_beans(self):
        """测试非Mock模式下，Mock Bean被过滤"""
        # 确保不在Mock模式
        assert not self.container.is_mock_mode()

        # 获取UserRepository，应返回非Mock实现
        repo = self.container.get_bean_by_type(UserRepository)
        assert not isinstance(repo, MockUserRepository)
        assert isinstance(repo, MySQLUserRepository)  # Primary非Mock实现

        # 获取所有实现，不应包含Mock
        all_repos = self.container.get_beans_by_type(UserRepository)
        assert len(all_repos) == 2  # MySQL + PostgreSQL
        assert all(not isinstance(r, MockUserRepository) for r in all_repos)

    def test_mock_mode_prioritizes_mock_beans(self):
        """测试Mock模式下，Mock Bean优先"""
        # 启用Mock模式
        self.container.enable_mock_mode()
        assert self.container.is_mock_mode()

        # 清除缓存以确保重新选择Bean
        self.container._singleton_instances.clear()

        # 获取UserRepository，应返回Mock实现
        repo = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo, MockUserRepository)
        assert repo.db_type == "mock"

        # 验证Mock数据
        user = repo.find_by_id(1)
        assert "Mock" in user["name"]
        assert user["source"] == "mock"

    def test_mock_mode_toggle(self):
        """测试Mock模式的切换"""
        # 初始：非Mock模式
        repo1 = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo1, MySQLUserRepository)

        # 启用Mock模式
        self.container.enable_mock_mode()
        self.container._singleton_instances.clear()

        repo2 = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo2, MockUserRepository)

        # 禁用Mock模式
        self.container.disable_mock_mode()
        self.container._singleton_instances.clear()

        repo3 = self.container.get_bean_by_type(UserRepository)
        assert isinstance(repo3, MySQLUserRepository)

    def test_get_all_beans_in_mock_mode(self):
        """测试Mock模式下获取所有Bean"""
        # 启用Mock模式
        self.container.enable_mock_mode()

        # 获取所有UserRepository实现
        all_repos = self.container.get_beans_by_type(UserRepository)

        # 应包含Mock实现，且Mock在最前面
        assert len(all_repos) == 3
        assert isinstance(all_repos[0], MockUserRepository)  # Mock优先
        assert isinstance(all_repos[1], MySQLUserRepository)  # Primary次之
        assert isinstance(all_repos[2], PostgreSQLUserRepository)


class TestBeanScopes:
    """测试Bean的不同作用域"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()
        PrototypeService.reset_counter()

    def test_singleton_scope(self):
        """测试Singleton作用域（默认）"""
        # 注册Singleton Bean
        self.container.register_bean(
            bean_type=MySQLUserRepository,
            bean_name="mysql_repo",
            scope=BeanScope.SINGLETON,
        )

        # 多次获取应返回同一实例
        repo1 = self.container.get_bean_by_type(MySQLUserRepository)
        repo2 = self.container.get_bean_by_type(MySQLUserRepository)
        repo3 = self.container.get_bean("mysql_repo")

        assert repo1 is repo2
        assert repo1 is repo3

        # 修改状态验证
        repo1.call_count = 100
        assert repo2.call_count == 100
        assert repo3.call_count == 100

    def test_prototype_scope(self):
        """测试Prototype作用域（每次创建新实例）"""
        # 注册Prototype Bean
        self.container.register_bean(
            bean_type=PrototypeService,
            bean_name="prototype_service",
            scope=BeanScope.PROTOTYPE,
        )

        # 多次获取应返回不同实例
        service1 = self.container.get_bean_by_type(PrototypeService)
        service2 = self.container.get_bean_by_type(PrototypeService)
        service3 = self.container.get_bean("prototype_service")

        assert service1 is not service2
        assert service1 is not service3
        assert service2 is not service3

        # 验证实例ID不同
        assert service1.instance_id == 1
        assert service2.instance_id == 2
        assert service3.instance_id == 3

        # 验证实例状态独立
        service1.add_data("data1")
        service2.add_data("data2")

        assert service1.get_data() == ["data1"]
        assert service2.get_data() == ["data2"]

    def test_factory_scope(self):
        """测试Factory作用域"""
        # 注册Factory Bean (register_factory默认就是FACTORY scope)
        self.container.register_factory(
            bean_type=DatabaseConnection,
            factory_method=create_database_connection,
            bean_name="db_connection",
        )

        # 获取Bean
        conn = self.container.get_bean_by_type(DatabaseConnection)

        # 验证Bean是通过工厂方法创建的
        assert isinstance(conn, DatabaseConnection)
        assert conn.host == "localhost"
        assert conn.port == 3306
        assert conn.database == "test_db"
        assert conn.connected


class TestFactoryBeans:
    """测试Factory Bean"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()

    def test_factory_bean_creation(self):
        """测试通过工厂方法创建Bean"""
        # 注册Factory
        self.container.register_factory(
            bean_type=DatabaseConnection,
            factory_method=create_database_connection,
            bean_name="db_connection",
        )

        # 获取Bean
        conn = self.container.get_bean_by_type(DatabaseConnection)

        # 验证Bean正确创建
        assert isinstance(conn, DatabaseConnection)
        assert conn.connected

        # 调用方法验证
        result = conn.execute("SELECT * FROM users")
        assert len(result) == 1
        assert "Executed" in result[0]["result"]

    def test_multiple_factories_for_same_type(self):
        """测试同一类型的多个Factory"""
        # 注册多个Factory
        self.container.register_factory(
            bean_type=DatabaseConnection,
            factory_method=create_database_connection,
            bean_name="db_connection",
            is_primary=True,
        )
        self.container.register_factory(
            bean_type=DatabaseConnection,
            factory_method=create_readonly_connection,
            bean_name="readonly_connection",
        )

        # 获取Primary Factory创建的Bean
        conn = self.container.get_bean_by_type(DatabaseConnection)
        assert conn.host == "localhost"

        # 通过名称获取另一个Factory创建的Bean
        readonly_conn = self.container.get_bean("readonly_connection")
        assert readonly_conn.host == "readonly.example.com"

    def test_factory_with_priority(self):
        """测试Factory Bean的优先级"""

        # 注册Regular Bean和Factory Bean
        def factory() -> CacheService:
            cache = RedisCacheService()
            cache.set("init_key", "init_value")
            return cache

        # Regular Bean
        self.container.register_bean(
            bean_type=MemoryCacheService, bean_name="memory_cache", is_primary=False
        )

        # Factory Bean (Primary)
        self.container.register_factory(
            bean_type=CacheService,
            factory_method=factory,
            bean_name="redis_cache",
            is_primary=True,
        )

        # 应返回Factory创建的Primary Bean
        cache = self.container.get_bean_by_type(CacheService)
        assert isinstance(cache, RedisCacheService)
        assert cache.get("init_key") == "init_value"


class TestDependencyInjection:
    """测试依赖注入"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()
        register_standard_beans(self.container)

    def test_constructor_injection(self):
        """测试构造函数依赖注入"""
        # 获取依赖
        repo = self.container.get_bean_by_type(UserRepository)

        # 手动注入依赖创建Service
        service = UserServiceImpl(repository=repo)

        # 验证依赖正确注入
        assert isinstance(service.repository, MySQLUserRepository)

        # 验证服务功能
        user = service.get_user(1)
        assert user["id"] == 1
        assert user["source"] == "mysql"

    def test_container_based_injection(self):
        """测试通过容器解析依赖"""
        # 创建Service时传入容器
        service = UserServiceImpl(container=self.container)

        # 验证依赖通过容器解析
        assert isinstance(service.repository, MySQLUserRepository)

        # 验证服务功能
        users = service.get_all_users()
        assert len(users) == 2
        assert all(u["source"] == "mysql" for u in users)

    def test_dependency_injection_in_mock_mode(self):
        """测试Mock模式下的依赖注入"""
        # 启用Mock模式
        self.container.enable_mock_mode()

        # 创建Service（应注入Mock依赖）
        service = UserServiceImpl(container=self.container)

        # 验证注入的是Mock依赖
        assert isinstance(service.repository, MockUserRepository)

        # 验证Mock数据
        user = service.get_user(1)
        assert "Mock" in user["name"]
        assert user["source"] == "mock"


class TestBeanOrdering:
    """测试Bean排序优先级"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()

    def test_ordering_primary_over_non_primary(self):
        """测试Primary优先于非Primary"""
        # 乱序注册
        self.container.register_bean(
            bean_type=SMSNotificationService, bean_name="sms", is_primary=False
        )
        self.container.register_bean(
            bean_type=EmailNotificationService, bean_name="email", is_primary=True
        )
        self.container.register_bean(
            bean_type=PushNotificationService, bean_name="push", is_primary=False
        )

        # 获取所有Bean，验证Primary在前
        services = self.container.get_beans_by_type(NotificationService)
        assert isinstance(services[0], EmailNotificationService)  # Primary

    def test_ordering_factory_over_singleton(self):
        """测试Factory优先于Singleton（同为Primary时）"""

        # 定义测试接口
        class TestService(ABC):
            pass

        class ServiceA(TestService):
            def __init__(self):
                self.type = "singleton"

        class ServiceB(TestService):
            def __init__(self):
                self.type = "factory"

        # 注册Singleton (Primary)
        self.container.register_bean(
            bean_type=ServiceA,
            bean_name="service_a",
            is_primary=True,
            scope=BeanScope.SINGLETON,
        )

        # 注册Factory (Primary) - register_factory默认就是FACTORY scope
        self.container.register_factory(
            bean_type=ServiceB,
            factory_method=lambda: ServiceB(),
            bean_name="service_b",
            is_primary=True,
        )

        # Factory应优先
        service = self.container.get_bean_by_type(TestService)
        assert isinstance(service, ServiceB)
        assert service.type == "factory"


class TestComplexScenarios:
    """测试复杂场景"""

    def setup_method(self):
        """每个测试前创建新容器"""
        self.container = DIContainer()
        register_standard_beans(self.container)

    def test_multiple_interface_implementations(self):
        """测试多个接口各有多个实现"""
        # 获取不同接口的Primary实现
        user_repo = self.container.get_bean_by_type(UserRepository)
        notification = self.container.get_bean_by_type(NotificationService)
        cache = self.container.get_bean_by_type(CacheService)

        # 验证各自的Primary实现
        assert isinstance(user_repo, MySQLUserRepository)
        assert isinstance(notification, EmailNotificationService)
        assert isinstance(cache, RedisCacheService)

    def test_get_all_beans_for_multiple_interfaces(self):
        """测试获取多个接口的所有实现"""
        # UserRepository: 2个实现（非Mock模式）
        user_repos = self.container.get_beans_by_type(UserRepository)
        assert len(user_repos) == 2

        # NotificationService: 3个实现
        notifications = self.container.get_beans_by_type(NotificationService)
        assert len(notifications) == 3

        # CacheService: 2个实现
        caches = self.container.get_beans_by_type(CacheService)
        assert len(caches) == 2

    def test_bean_lifecycle_and_state(self):
        """测试Bean的生命周期和状态管理"""
        # 获取Singleton Bean
        repo = self.container.get_bean_by_type(UserRepository)

        # 修改状态
        repo.find_by_id(1)
        repo.find_all()
        assert repo.call_count == 2

        # 再次获取，应该是同一实例，状态保持
        repo2 = self.container.get_bean_by_type(UserRepository)
        assert repo2.call_count == 2

        # 调用方法，状态继续累加
        repo2.find_by_id(2)
        assert repo.call_count == 3
        assert repo2.call_count == 3


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s", "--tb=short"])
