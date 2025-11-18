# 智能记忆系统开发指南

本文档为开发者提供接口定义、Mock实现和解耦开发的详细指导。

## 📋 目录

- [接口定义和实现](#接口定义和实现)
- [Mock实现和解耦开发](#mock实现和解耦开发)
- [依赖注入最佳实践](#依赖注入最佳实践)
- [开发环境配置](#开发环境配置)

## 🔧 接口定义和实现

### 1. 定义抽象接口

使用Python的抽象基类（ABC）定义清晰的接口：

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class UserRepository(ABC):
    """用户存储接口"""
    
    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        """根据ID查找用户"""
        pass
    
    @abstractmethod
    def save(self, user: Dict) -> Dict:
        """保存用户"""
        pass
    
    @abstractmethod
    def find_by_email(self, email: str) -> Optional[Dict]:
        """根据邮箱查找用户"""
        pass

class NotificationService(ABC):
    """通知服务接口"""
    
    @abstractmethod
    async def send_notification(self, user_id: int, message: str) -> bool:
        """发送通知"""
        pass
```

### 2. 实现具体类

使用依赖注入装饰器标记实现类：

```python
from core.di.decorators import repository, service

@repository("mysql_user_repo")
class MySQLUserRepository(UserRepository):
    """MySQL用户存储实现"""
    
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        # 实际数据库查询逻辑
        return {"id": user_id, "name": f"User {user_id}", "source": "mysql"}
    
    def save(self, user: Dict) -> Dict:
        # 实际保存逻辑
        return {**user, "id": 123, "created_at": "2024-01-01"}
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        # 实际查询逻辑
        return {"id": 456, "email": email, "source": "mysql"}

@repository("redis_user_repo") 
class RedisUserRepository(UserRepository):
    """Redis用户存储实现（缓存层）"""
    
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        # Redis缓存查询逻辑
        return {"id": user_id, "name": f"Cached User {user_id}", "source": "redis"}
    
    def save(self, user: Dict) -> Dict:
        # Redis缓存保存逻辑
        return {**user, "cached": True}
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        # Redis缓存查询逻辑
        return None  # 缓存中没有

@service("email_notification")
class EmailNotificationService(NotificationService):
    """邮件通知服务实现"""
    
    async def send_notification(self, user_id: int, message: str) -> bool:
        # 实际发送邮件逻辑
        print(f"📧 发送邮件给用户 {user_id}: {message}")
        return True
```

### 3. 设置Primary实现

当有多个实现时，使用`primary=True`标记默认实现：

```python
@repository("primary_user_repo", primary=True)
class PrimaryUserRepository(UserRepository):
    """主要用户存储实现"""
    
    def __init__(self):
        # 可以组合多个实现
        self.mysql_repo = MySQLUserRepository()
        self.redis_repo = RedisUserRepository()
    
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        # 先查缓存，再查数据库
        user = self.redis_repo.find_by_id(user_id)
        if user:
            return user
        return self.mysql_repo.find_by_id(user_id)
    
    def save(self, user: Dict) -> Dict:
        # 保存到数据库并更新缓存
        saved_user = self.mysql_repo.save(user)
        self.redis_repo.save(saved_user)
        return saved_user
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        return self.mysql_repo.find_by_email(email)
```

## 🧪 Mock实现和解耦开发

### 1. 定义Mock实现

使用`@mock_impl`装饰器定义Mock实现：

```python
from core.di.decorators import mock_impl

@mock_impl("mock_user_repo")
class MockUserRepository(UserRepository):
    """Mock用户存储实现"""
    
    def __init__(self):
        # 内存中的模拟数据
        self.users = {
            1: {"id": 1, "name": "Mock User 1", "email": "user1@mock.com"},
            2: {"id": 2, "name": "Mock User 2", "email": "user2@mock.com"}
        }
        self.next_id = 3
    
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        return self.users.get(user_id)
    
    def save(self, user: Dict) -> Dict:
        if "id" not in user:
            user["id"] = self.next_id
            self.next_id += 1
        self.users[user["id"]] = user
        return user
    
    def find_by_email(self, email: str) -> Optional[Dict]:
        for user in self.users.values():
            if user.get("email") == email:
                return user
        return None

@mock_impl("mock_notification")
class MockNotificationService(NotificationService):
    """Mock通知服务实现"""
    
    def __init__(self):
        self.sent_messages = []  # 记录发送的消息用于测试验证
    
    async def send_notification(self, user_id: int, message: str) -> bool:
        # 模拟发送，实际只是记录
        notification = {
            "user_id": user_id,
            "message": message,
            "timestamp": "2024-01-01T10:00:00"
        }
        self.sent_messages.append(notification)
        print(f"🧪 Mock发送通知: {notification}")
        return True
    
    def get_sent_messages(self) -> List[Dict]:
        """获取已发送的消息（测试用）"""
        return self.sent_messages.copy()
```

### 2. Mock模式开关控制

#### 2.1 环境变量控制

在环境变量文件（`.env`）中配置：

```bash
# 开发环境配置
MOCK_MODE=true

# 生产环境配置
# MOCK_MODE=false
```

#### 2.2 代码中动态切换

```python
import os
from core.di.utils import enable_mock_mode, disable_mock_mode, get_bean_by_type

def setup_mock_mode():
    """根据环境变量设置Mock模式"""
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        enable_mock_mode()
        print("🧪 已启用Mock模式")
    else:
        disable_mock_mode()
        print("🔧 使用真实实现")

# 在应用启动时调用
def initialize_app():
    setup_mock_mode()
    
    # 现在获取的实现会根据Mock模式自动切换
    user_service = get_bean_by_type(UserService)
    return user_service
```

#### 2.3 启动时自动控制

应用启动时会自动检查`MOCK_MODE`环境变量：

```python
# 在run.py中的实现
if os.getenv("MOCK_MODE") and os.getenv("MOCK_MODE").lower() == "true":
    enable_mock_mode()
    logger.info("🚀 启用Mock模式")
else:
    logger.info("🚀 禁用Mock模式")
```

### 3. 条件Mock实现

根据不同条件使用不同的Mock实现：

```python
@mock_impl("mock_fast_notification")
class FastMockNotificationService(NotificationService):
    """快速Mock通知（测试用）"""
    
    async def send_notification(self, user_id: int, message: str) -> bool:
        print(f"⚡ 快速Mock通知: 用户{user_id} - {message}")
        return True

@mock_impl("mock_slow_notification") 
class SlowMockNotificationService(NotificationService):
    """慢速Mock通知（性能测试用）"""
    
    async def send_notification(self, user_id: int, message: str) -> bool:
        import asyncio
        await asyncio.sleep(0.1)  # 模拟网络延迟
        print(f"🐌 慢速Mock通知: 用户{user_id} - {message}")
        return True

# 根据测试类型选择Mock实现
def setup_test_environment(test_type: str):
    enable_mock_mode()
    
    if test_type == "performance":
        # 性能测试使用慢速Mock
        from core.di.utils import register_bean
        slow_mock = SlowMockNotificationService()
        register_bean(NotificationService, slow_mock, "mock_notification")
    else:
        # 普通测试使用快速Mock
        fast_mock = FastMockNotificationService()
        register_bean(NotificationService, fast_mock, "mock_notification")
```

## 🏗️ 依赖注入最佳实践

### 1. 接口设计原则

- **单一职责**：每个接口只负责一个明确的职责
- **接口隔离**：客户端不应依赖它不需要的接口
- **依赖倒置**：高层模块不应依赖低层模块，都应依赖抽象

```python
# 好的设计：职责明确
class UserRepository(ABC):
    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        pass

class UserValidator(ABC):
    @abstractmethod
    def validate(self, user: Dict) -> bool:
        pass

# 避免的设计：职责混合
class UserService(ABC):  # 不推荐：混合了存储和验证职责
    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def validate(self, user: Dict) -> bool:
        pass
```

### 2. 装饰器使用规范

```python
# 从具体的装饰器模块导入
from core.di.decorators import repository, service, component, mock_impl, factory

# 数据访问层
@repository("user_repository")
class UserRepositoryImpl(UserRepository):
    pass

# 业务服务层
@service("user_service")
class UserService:
    pass

# 通用组件
@component("config_manager")
class ConfigManager:
    pass

# Mock实现
@mock_impl("mock_external_api")
class MockExternalApiClient(ExternalApiClient):
    pass

# 工厂方法
@factory(DatabaseConnection, "db_connection")
def create_database_connection() -> DatabaseConnection:
    config = load_config()
    return DatabaseConnection(config.db_url)
```

### 3. 循环依赖处理

使用延迟注入避免循环依赖：

```python
from core.di.decorators import service
from core.di.utils import lazy_inject

@service("order_service")
class OrderService:
    def __init__(self):
        # 延迟获取依赖，避免循环依赖
        self.user_service_lazy = lazy_inject(UserService)
        self.payment_service_lazy = lazy_inject(PaymentService)
    
    def create_order(self, order_data: Dict) -> Dict:
        user_service = self.user_service_lazy()  # 调用时才获取
        payment_service = self.payment_service_lazy()
        
        # 业务逻辑
        user = user_service.get_user(order_data["user_id"])
        payment_result = payment_service.process_payment(order_data["amount"])
        
        return {"order_id": 123, "status": "created"}
```

## ⚙️ 开发环境配置

**注意**：开始开发前，请先运行 `pre-commit install` 安装代码格式检查工具，保证代码风格一致性。

### 1. 环境变量配置

创建`.env.development`文件：

```bash
# 开发环境配置
ENVIRONMENT=development
DEBUG=true

# Mock模式配置
MOCK_MODE=true

# 日志配置
LOG_LEVEL=DEBUG
LOG_FORMAT=detailed

# 外部服务配置（开发环境使用测试地址）
EXTERNAL_API_URL=https://api-test.example.com
DATABASE_URL=postgresql://dev:password@localhost:5432/memsys_dev
REDIS_URL=redis://localhost:6379/0
```

### 2. 开发脚本模板

对于需要运行的开发脚本，在脚本开头添加环境初始化：

```python
#!/usr/bin/env python3
"""
开发脚本模板 - 数据处理/测试脚本等
"""
import os

# ============= 开发环境初始化 (必须在最上面) =============
# 1. 设置环境变量和Python路径
from common_utils.load_env import setup_environment
setup_environment(load_env_file_name=".env.development", check_env_var="GEMINI_API_KEY")

# 2. 启用Mock模式（开发环境默认启用）
from core.di.utils import enable_mock_mode
if os.getenv("MOCK_MODE", "true").lower() == "true":
    enable_mock_mode()
    print("🧪 开发脚本：已启用Mock模式")

# 3. 初始化依赖注入
from application_startup import setup_all
setup_all()
# ================================================

# 现在可以正常导入和使用项目模块
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)

def main():
    """脚本主逻辑"""
    logger.info("🚀 开发脚本开始执行")
    
    # 示例：使用依赖注入获取服务
    # user_service = get_bean_by_type(UserService)
    # result = user_service.process_data()
    
    # 你的脚本逻辑...
    
    logger.info("✅ 开发脚本执行完成")

if __name__ == "__main__":
    main()
```

#### 实际使用示例

```python
#!/usr/bin/env python3
"""
用户数据迁移脚本
"""
import os

# ============= 开发环境初始化 =============
from common_utils.load_env import setup_environment
setup_environment(load_env_file_name=".env.development")

from core.di.utils import enable_mock_mode
if os.getenv("MOCK_MODE", "true").lower() == "true":
    enable_mock_mode()
    print("🧪 数据迁移脚本：使用Mock数据")

from application_startup import setup_all
setup_all()
# =======================================

from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)

def migrate_user_data():
    """迁移用户数据"""
    logger.info("开始迁移用户数据...")
    
    # 获取服务（会自动使用Mock实现，如果启用了Mock模式）
    user_service = get_bean_by_type(UserService)
    
    # 处理数据迁移
    users = user_service.get_all_users()
    for user in users:
        # 迁移逻辑...
        logger.info(f"迁移用户: {user['name']}")
    
    logger.info("用户数据迁移完成")

if __name__ == "__main__":
    migrate_user_data()
```

### 3. 开发启动方式

#### 运行开发脚本
```bash
# 进入src目录
cd src

# 运行数据处理脚本
python your_dev_script.py

# 运行迁移脚本
python migrate_data.py

# 运行测试脚本
python test_service.py
```

#### 启动开发服务
```bash
# 使用开发环境启动Web服务
python run.py --env-file .env.development

# 或者设置环境变量后启动
export MOCK_MODE=true
python run.py
```

#### VS Code调试配置

在VS Code的`launch.json`中添加开发配置：

```json
{
    "name": "开发模式启动",
    "type": "debugpy",
    "request": "launch",
    "env": {
        "PYTHONPATH": "${workspaceFolder}/src"
    },
    "envFile": "${workspaceFolder}/.env.development",
    "cwd": "${workspaceFolder}/src",
    "python": "${workspaceFolder}/.venv/bin/python",
    "program": "dev_run.py",
    "console": "integratedTerminal",
    "justMyCode": false
}
```

### 4. Mock模式验证

启动应用后，可以通过日志确认Mock模式状态：

```bash
# 启动应用时会显示Mock模式状态
python run.py

# 输出示例：
# 🚀 启用Mock模式  (当MOCK_MODE=true时)
# 🚀 禁用Mock模式  (当MOCK_MODE=false或未设置时)
```

## 📝 实际开发示例

### 完整的开发流程示例

```python
from core.di.decorators import service, mock_impl

# 1. 定义接口
class PaymentProcessor(ABC):
    @abstractmethod
    async def process_payment(self, amount: float, payment_method: str) -> Dict:
        pass

# 2. 实现真实服务
@service("stripe_payment")
class StripePaymentProcessor(PaymentProcessor):
    async def process_payment(self, amount: float, payment_method: str) -> Dict:
        # 真实的Stripe API调用
        return {"transaction_id": "stripe_123", "status": "success"}

# 3. 实现Mock服务
@mock_impl("mock_payment")
class MockPaymentProcessor(PaymentProcessor):
    async def process_payment(self, amount: float, payment_method: str) -> Dict:
        # Mock实现，用于开发和测试
        return {"transaction_id": "mock_123", "status": "success"}

# 4. 业务服务使用接口
@service("order_service")
class OrderService:
    def __init__(self, payment_processor: PaymentProcessor):
        self.payment_processor = payment_processor
    
    async def place_order(self, order_data: Dict) -> Dict:
        # 处理支付
        payment_result = await self.payment_processor.process_payment(
            order_data["amount"], 
            order_data["payment_method"]
        )
        
        if payment_result["status"] == "success":
            return {"order_id": 456, "status": "confirmed"}
        else:
            return {"error": "Payment failed"}

# 5. 开发时使用
def development_workflow():
    from core.di.utils import enable_mock_mode, get_bean_by_type
    
    # 启用Mock模式进行开发
    enable_mock_mode()
    
    # 获取服务（自动使用Mock实现）
    order_service = get_bean_by_type(OrderService)
    
    # 测试业务逻辑，无需真实支付
    order_data = {
        "amount": 99.99,
        "payment_method": "credit_card"
    }
    
    result = await order_service.place_order(order_data)
    print(f"订单结果: {result}")

# 6. 生产环境使用
def production_workflow():
    from core.di.utils import disable_mock_mode, get_bean_by_type
    
    # 禁用Mock模式使用真实服务
    disable_mock_mode()
    
    # 获取服务（自动使用真实实现）
    order_service = get_bean_by_type(OrderService)
    
    # 真实业务处理
    result = await order_service.place_order(order_data)
    print(f"真实订单结果: {result}")
```

通过这种方式，开发者可以：

1. **并行开发**：前端和后端可以同时开发，后端使用Mock数据
2. **快速测试**：无需搭建完整的外部服务环境
3. **解耦开发**：各个模块可以独立开发和测试
4. **灵活切换**：通过简单的配置切换Mock和真实实现

这种架构大大提高了开发效率和代码质量，同时保持了系统的可测试性和可维护性。
