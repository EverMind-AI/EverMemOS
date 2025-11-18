# -*- coding: utf-8 -*-
"""
DI测试 Fixtures

提供测试用的接口、实现类、Mock类等
这些类可以被其他测试文件导入使用
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from core.di.container import DIContainer
from core.di.bean_definition import BeanScope


# ==================== 用户服务相关 ====================


class UserRepository(ABC):
    """用户仓储接口"""

    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[dict]:
        """根据ID查找用户"""
        pass

    @abstractmethod
    def find_all(self) -> List[dict]:
        """查找所有用户"""
        pass


class MySQLUserRepository(UserRepository):
    """MySQL用户仓储实现（Primary）"""

    def __init__(self):
        self.db_type = "mysql"
        self.call_count = 0

    def find_by_id(self, user_id: int) -> Optional[dict]:
        self.call_count += 1
        return {"id": user_id, "name": f"User{user_id}", "source": "mysql"}

    def find_all(self) -> List[dict]:
        self.call_count += 1
        return [
            {"id": 1, "name": "User1", "source": "mysql"},
            {"id": 2, "name": "User2", "source": "mysql"},
        ]


class PostgreSQLUserRepository(UserRepository):
    """PostgreSQL用户仓储实现（非Primary）"""

    def __init__(self):
        self.db_type = "postgres"
        self.call_count = 0

    def find_by_id(self, user_id: int) -> Optional[dict]:
        self.call_count += 1
        return {"id": user_id, "name": f"User{user_id}", "source": "postgres"}

    def find_all(self) -> List[dict]:
        self.call_count += 1
        return [{"id": 1, "name": "User1", "source": "postgres"}]


class MockUserRepository(UserRepository):
    """Mock用户仓储实现"""

    def __init__(self):
        self.db_type = "mock"
        self.call_count = 0

    def find_by_id(self, user_id: int) -> Optional[dict]:
        self.call_count += 1
        return {"id": user_id, "name": f"MockUser{user_id}", "source": "mock"}

    def find_all(self) -> List[dict]:
        self.call_count += 1
        return [{"id": 999, "name": "MockUser", "source": "mock"}]


class UserService(ABC):
    """用户服务接口"""

    @abstractmethod
    def get_user(self, user_id: int) -> Optional[dict]:
        """获取用户"""
        pass

    @abstractmethod
    def get_all_users(self) -> List[dict]:
        """获取所有用户"""
        pass


class UserServiceImpl(UserService):
    """用户服务实现"""

    def __init__(
        self, repository: UserRepository = None, container: DIContainer = None
    ):
        # 支持两种注入方式：构造函数注入或通过容器获取
        if repository:
            self.repository = repository
        elif container:
            self.repository = container.get_bean_by_type(UserRepository)
        else:
            raise ValueError("Must provide either repository or container")
        self.call_count = 0

    def get_user(self, user_id: int) -> Optional[dict]:
        self.call_count += 1
        return self.repository.find_by_id(user_id)

    def get_all_users(self) -> List[dict]:
        self.call_count += 1
        return self.repository.find_all()


# ==================== 通知服务相关 ====================


class NotificationService(ABC):
    """通知服务接口"""

    @abstractmethod
    def send(self, message: str, recipient: str) -> bool:
        """发送通知"""
        pass


class EmailNotificationService(NotificationService):
    """邮件通知服务实现（Primary）"""

    def __init__(self):
        self.sent_messages = []

    def send(self, message: str, recipient: str) -> bool:
        self.sent_messages.append(
            {"message": message, "recipient": recipient, "type": "email"}
        )
        return True


class SMSNotificationService(NotificationService):
    """短信通知服务实现（非Primary）"""

    def __init__(self):
        self.sent_messages = []

    def send(self, message: str, recipient: str) -> bool:
        self.sent_messages.append(
            {"message": message, "recipient": recipient, "type": "sms"}
        )
        return True


class PushNotificationService(NotificationService):
    """推送通知服务实现"""

    def __init__(self):
        self.sent_messages = []

    def send(self, message: str, recipient: str) -> bool:
        self.sent_messages.append(
            {"message": message, "recipient": recipient, "type": "push"}
        )
        return True


# ==================== 邮件服务相关 ====================


class EmailService(ABC):
    """邮件服务接口"""

    @abstractmethod
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """发送邮件"""
        pass


class SMTPEmailService(EmailService):
    """SMTP邮件服务实现"""

    def __init__(self):
        self.host = "smtp.example.com"
        self.port = 587
        self.sent_emails = []

    def send_email(self, to: str, subject: str, body: str) -> bool:
        self.sent_emails.append({"to": to, "subject": subject, "body": body})
        return True


# ==================== 数据库连接相关 ====================


class DatabaseConnection:
    """数据库连接类"""

    def __init__(self, host: str, port: int, database: str):
        self.host = host
        self.port = port
        self.database = database
        self.connected = True

    def execute(self, sql: str) -> List[dict]:
        return [{"result": f"Executed: {sql}"}]

    def close(self):
        self.connected = False


def create_database_connection() -> DatabaseConnection:
    """创建数据库连接的工厂方法"""
    return DatabaseConnection(host="localhost", port=3306, database="test_db")


def create_readonly_connection() -> DatabaseConnection:
    """创建只读数据库连接的工厂方法"""
    return DatabaseConnection(
        host="readonly.example.com", port=3306, database="test_db"
    )


# ==================== Prototype Scope 测试类 ====================


class PrototypeService:
    """原型作用域服务（每次获取都创建新实例）"""

    instance_counter = 0  # 类级别计数器

    def __init__(self):
        PrototypeService.instance_counter += 1
        self.instance_id = PrototypeService.instance_counter
        self.data = []

    def add_data(self, value: str):
        self.data.append(value)

    def get_data(self) -> List[str]:
        return self.data

    @classmethod
    def reset_counter(cls):
        """重置计数器（用于测试）"""
        cls.instance_counter = 0


# ==================== 缓存服务相关 ====================


class CacheService(ABC):
    """缓存服务接口"""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """获取缓存"""
        pass

    @abstractmethod
    def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """设置缓存"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存"""
        pass


class RedisCacheService(CacheService):
    """Redis缓存服务实现（Primary）"""

    def __init__(self):
        self.storage = {}
        self.cache_type = "redis"

    def get(self, key: str) -> Optional[str]:
        return self.storage.get(key)

    def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        self.storage[key] = value
        return True

    def delete(self, key: str) -> bool:
        if key in self.storage:
            del self.storage[key]
            return True
        return False


class MemoryCacheService(CacheService):
    """内存缓存服务实现"""

    def __init__(self):
        self.storage = {}
        self.cache_type = "memory"

    def get(self, key: str) -> Optional[str]:
        return self.storage.get(key)

    def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        self.storage[key] = value
        return True

    def delete(self, key: str) -> bool:
        if key in self.storage:
            del self.storage[key]
            return True
        return False


# ==================== 工具辅助函数 ====================


def register_standard_beans(container: DIContainer):
    """注册标准测试Bean到容器

    Args:
        container: DI容器实例
    """
    # 注册UserRepository实现
    container.register_bean(
        bean_type=MySQLUserRepository, bean_name="mysql_user_repo", is_primary=True
    )
    container.register_bean(
        bean_type=PostgreSQLUserRepository,
        bean_name="postgres_user_repo",
        is_primary=False,
    )
    container.register_bean(
        bean_type=MockUserRepository, bean_name="mock_user_repo", is_mock=True
    )

    # 注册NotificationService实现
    container.register_bean(
        bean_type=EmailNotificationService,
        bean_name="email_notification",
        is_primary=True,
    )
    container.register_bean(
        bean_type=SMSNotificationService, bean_name="sms_notification"
    )
    container.register_bean(
        bean_type=PushNotificationService, bean_name="push_notification"
    )

    # 注册CacheService实现
    container.register_bean(
        bean_type=RedisCacheService, bean_name="redis_cache", is_primary=True
    )
    container.register_bean(bean_type=MemoryCacheService, bean_name="memory_cache")

    # 注册EmailService实现
    container.register_bean(bean_type=SMTPEmailService, bean_name="smtp_email_service")

    # 注册Factory Bean
    container.register_factory(
        bean_type=DatabaseConnection,
        factory_method=create_database_connection,
        bean_name="db_connection",
    )

    # 注册Prototype Bean
    container.register_bean(
        bean_type=PrototypeService,
        bean_name="prototype_service",
        scope=BeanScope.PROTOTYPE,
    )
