# 开发规范

本文档介绍项目开发过程中的各项规范和最佳实践，帮助团队保持代码质量和协作效率。

---

## 🚀 TL;DR (核心原则)

### 新人上手（3 步启动）
```bash
uv sync --group dev-full    # 同步依赖
pre-commit install           # 安装代码检查钩子
```

### 核心约定

**📦 依赖管理**  
使用 `uv add/remove` 管理依赖，避免直接 `pip install`，以保持依赖锁文件的一致性

**🎨 代码风格**  
提交时自动运行 pre-commit 检查（black/ruff/isort），保持代码风格统一

**⚡️ 全异步架构**  
单 Event Loop，I/O 操作使用 `async/await`，线程/进程使用前建议与开发负责人讨论

**🚫 禁止循环中 I/O**  
禁止在 for 循环中进行数据库访问和 API 调用等 I/O 操作，使用批量操作替代

**🕐 时区意识**  
所有时间字段必须携带时区信息，不带时区的输入统一视为上海时区（Asia/Shanghai，UTC+8）。禁止使用 `datetime.datetime.now()`，必须使用 `common_utils/datetime_utils.py` 中的工具函数

**📥 导入规范**  
- PYTHONPATH 管理：项目模块导入起始路径（src/tests/demo 等）需要统一管理，变更前请与开发负责人沟通  
- 优先使用绝对导入（如 `from core.memory import MemoryManager`），避免相对导入（如 `from ...core import`）

**📝 __init__.py 规范**  
不建议在 `__init__.py` 中编写任何代码，保持空文件即可

**🌿 分支规范**  
`dev` 日常开发，`release/YYMMDD` 定版发布，`long/xxx` 长期功能开发，`hotfix` 紧急修复

**🔀 分支合并统一处理**  
`long/xxx` 合并到 `dev`、`dev` 切出 `release`、`release` 合并回 `dev` 需由开发或运维负责人统一处理

**🔍 Code Review 要求**  
增加数据迁移脚本、依赖三方包新增或者升降级、基建框架代码改动、合并 发布release分支 必须走 MR 流程

**💾 数据迁移规范**  
涉及数据修复或 Schema 迁移的新功能，尽早与研发、运维讨论方案可行性和实施时间安排

**🏛️ 数据访问规范**  
所有数据库、搜索引擎等外部存储的读写操作必须收敛到 infra 层的 repository 方法中，禁止在业务层直接调用外部仓储

### 📖 快速导航

- 不知道怎么装依赖？→ [依赖管理规范](#依赖管理规范)
- 需要数据库/中间件配置？→ [开发环境配置规范](#开发环境配置规范)
- 提交前总报错？→ [代码风格规范](#代码风格规范)  
- 不确定能不能用线程？→ [异步编程规范](#异步编程规范)
- 循环中能做数据库查询吗？→ [禁止在 for 循环中进行 I/O 操作](#7-禁止在-for-循环中进行-io-操作-)
- 时间字段怎么处理？→ [时区意识规范](#时区意识规范)
- 数据库查询应该写在哪？→ [数据访问规范](#数据访问规范)
- 导入路径报错？→ [导入规范](#导入规范)
- 不知道切什么分支？→ [分支管理规范](#分支管理规范)
- 需要提 MR？→ [Code Review 流程](#code-review-流程)
- 需要数据迁移？→ [数据迁移与 Schema 变更流程](#数据迁移与-schema-变更流程)

---

## 📋 目录

- [TL;DR (快速上手)](#tldr-快速上手)
- [依赖管理规范](#依赖管理规范)
- [开发环境配置规范](#开发环境配置规范)
- [代码风格规范](#代码风格规范)
- [异步编程规范](#异步编程规范)
- [时区意识规范](#时区意识规范)
- [数据访问规范](#数据访问规范)
- [导入规范](#导入规范)
  - [PYTHONPATH 管理](#pythonpath-管理)
  - [优先使用绝对导入](#优先使用绝对导入)
  - [__init__.py 使用规范](#__init__py-使用规范)
- [分支管理规范](#分支管理规范)
- [Code Review 流程](#code-review-流程)
  - [数据迁移与 Schema 变更流程](#数据迁移与-schema-变更流程)

---

## 📦 依赖管理规范

### 使用 uv 进行依赖管理

**💡 重要提示：推荐使用 uv 管理依赖**

项目使用 `uv` 作为依赖管理工具，建议避免使用 `pip install` 直接安装包，原因如下：

- 依赖版本可能不一致
- `uv.lock` 文件无法自动更新
- 团队成员环境可能产生差异
- 可能影响生产环境部署

### 正确的操作方式

#### 1. 安装/同步依赖

```bash
# 同步所有依赖（首次安装或更新后）
uv sync --group dev-full

```

#### 2. 添加新依赖

```bash
# 添加生产依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 指定版本
uv add <package-name>==<version>
```

#### 3. 移除依赖

```bash
uv remove <package-name>
```

#### 4. 更新依赖

```bash
# 更新所有依赖
uv sync --upgrade

# 更新指定依赖
uv add <package-name> --upgrade
```

### 相关文档

详细的依赖管理指南请参考：[project_deps_manage.md](./project_deps_manage.md)

---

## 🔧 开发环境配置规范

### 环境配置说明

项目依赖多种数据库和中间件，为了保证开发环境的一致性和安全性，这些配置由运维团队统一管理和分发。

#### 涉及的配置项

开发环境通常需要以下配置：

**数据库配置**
- MongoDB 连接信息
- PostgreSQL 连接信息
- Redis 连接信息

**中间件配置**
- Kafka 连接配置
- ElasticSearch 连接配置
- 其他消息队列或缓存服务

**第三方服务配置**
- API 密钥和访问凭证
- 对象存储配置
- 其他外部服务凭证

### 如何获取配置

#### 1. 新人入职

新加入项目的开发者，请按以下流程获取配置：

1. **联系运维负责人**（见文档末尾联系人信息）
2. **说明需求**：
   - 你的姓名和角色
   - 需要的环境（开发环境/测试环境）
   - 需要访问的具体服务
3. **接收配置**：运维负责人会提供配置文件或环境变量
4. **本地配置**：将配置信息放入项目的 `config.json` 或 `.env` 文件（注意：这些文件已在 `.gitignore` 中，不会提交到代码库）

#### 2. 配置文件位置

```bash
# 项目根目录下的配置文件（请勿提交到 git）
config.json          # 主要配置文件
.env                 # 环境变量配置
env.template         # 配置模板（可参考但需要填入真实值）
```

#### 3. 环境变量示例

参考 `env.template` 文件，你的 `.env` 文件通常包含以下类型的配置：

```bash
# MongoDB
MONGODB_URI=mongodb://...
MONGODB_DATABASE=...

# Redis
REDIS_HOST=...
REDIS_PORT=...
REDIS_PASSWORD=...

# Kafka
KAFKA_BOOTSTRAP_SERVERS=...

# ElasticSearch
ES_HOST=...
ES_PORT=...
```

### 配置管理注意事项

#### ⚠️ 安全规范

1. **禁止提交敏感配置**
   - 所有包含密码、密钥、Token 的配置文件不得提交到 git
   - 提交前检查 `.gitignore` 是否包含配置文件
   - 使用 pre-commit hook 可以帮助检测敏感信息

2. **配置文件权限**
   - 本地配置文件建议设置适当的权限（仅当前用户可读）
   - 不要在公共场所（如聊天记录、文档）直接粘贴配置内容

3. **配置更新通知**
   - 如配置有更新，运维团队会通知相关开发者
   - 收到通知后及时更新本地配置

#### 🔄 配置变更流程

如果需要：
- 新增配置项
- 修改配置结构
- 新增环境或服务

**建议按以下流程操作**：

1. **与开发负责人讨论**：确认配置变更的必要性和影响范围
2. **联系运维负责人**：说明配置需求和变更原因
3. **更新配置模板**：更新 `env.template` 和相关文档
4. **团队通知**：通知所有开发者同步更新本地配置

#### 📝 配置问题排查

**常见问题**：

1. **连接失败**
   - 检查网络连接（是否在公司网络或 VPN）
   - 确认配置信息是否正确
   - 联系运维负责人确认服务状态

2. **权限不足**
   - 确认账号是否已授权
   - 联系运维负责人申请相应权限

3. **配置过期**
   - 定期检查配置是否需要更新
   - 关注团队通知中的配置变更信息

### 不同环境说明

| 环境 | 用途 | 配置来源 | 备注 |
|------|------|----------|------|
| **开发环境** | 本地开发和调试 | 运维提供 | 通常连接到开发数据库，数据可随意测试 |
| **测试环境** | 集成测试和功能测试 | 自动部署配置 | 连接到测试数据库，定期重置数据 |
| **生产环境** | 正式运行的服务 | 运维严格管控 | 仅运维和授权人员可访问 |

**注意**：开发者通常只需要开发环境配置，测试环境和生产环境的配置由 CI/CD 和运维团队管理。

---

## 🎨 代码风格规范

### Pre-commit Hook 配置

项目使用 `pre-commit` 统一代码风格，建议在首次克隆项目后安装 pre-commit hook。

#### 安装步骤

```bash
# 1. 确保已同步开发依赖
uv sync --dev

# 2. 安装 pre-commit hook
pre-commit install
```

#### 作用

Pre-commit hook 会在每次提交前自动执行以下检查：

- **代码格式化**：使用 black/ruff 格式化 Python 代码
- **Import 排序**：使用 isort 排序导入语句
- **代码检查**：使用 ruff/flake8 进行代码质量检查
- **类型检查**：使用 pyright/mypy 进行类型检查
- **YAML/JSON 格式**：检查配置文件格式
- **尾随空格**：移除文件末尾多余空格

#### 手动运行检查

```bash
# 对所有文件运行检查
pre-commit run --all-files

# 对暂存的文件运行检查
pre-commit run
```

---

## ⚡️ 异步编程规范

### 全异步架构原则

项目采用 **全异步架构**，基于以下原则：

#### 1. 单 Event Loop 原则

- **整个应用使用一个主 Event Loop**
- 避免在代码中创建新的 Event Loop（`asyncio.new_event_loop()`）
- 避免在异步上下文中使用 `asyncio.run()` 启动新循环

#### 2. 关于线程和进程的使用 ⚠️

**💡 重要提示：慎用多线程和多进程**

项目基于单 Event Loop 的全异步架构，建议避免以下操作：

```python
# ❌ 不推荐：创建线程
import threading
thread = threading.Thread(target=some_function)
thread.start()

# ❌ 不推荐：使用线程池（除非特殊情况）
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

# ❌ 不推荐：创建进程
import multiprocessing
process = multiprocessing.Process(target=some_function)
process.start()

# ❌ 不推荐：使用进程池
from concurrent.futures import ProcessPoolExecutor
executor = ProcessPoolExecutor()
```

**为什么不推荐？**
- 可能破坏单 Event Loop 架构，产生并发问题
- 线程安全问题较复杂，容易引入竞态条件
- 资源管理难度较高，可能造成资源泄漏
- 可能影响异步上下文（contextvars）的正常工作
- 调试难度会增加，堆栈追踪较复杂

**特殊场景处理**

如果确实需要使用线程或进程（例如 CPU 密集型计算、调用不支持异步的第三方库），建议：

1. **提前与开发负责人讨论方案**
2. 说明为什么异步方案无法满足需求
3. 提供资源管理方案（确保线程/进程正确关闭）
4. 通过 Code Review

**少数允许的场景示例**：

```python
# ✅ 特殊场景：调用不支持异步的同步库（经讨论后）
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 全局共享的线程池，限制最大线程数
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

async def call_sync_library(data):
    """调用不支持异步的第三方库（已与负责人确认）"""
    loop = asyncio.get_event_loop()
    # 在线程池中运行，避免阻塞主循环
    result = await loop.run_in_executor(
        _EXECUTOR,
        sync_blocking_function,
        data
    )
    return result
```

#### 3. 异步函数定义

I/O 操作建议使用异步函数：

```python
# ✅ 正确：异步函数
async def fetch_user_data(user_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/users/{user_id}")
        return response.json()

# ❌ 错误：同步 I/O
def fetch_user_data(user_id: str) -> dict:
    response = requests.get(f"/users/{user_id}")
    return response.json()
```

#### 4. 数据库操作

```python
# ✅ 正确：使用异步数据库驱动
from motor.motor_asyncio import AsyncIOMotorClient

async def get_user(db, user_id: str):
    return await db.users.find_one({"_id": user_id})

# ❌ 错误：使用同步驱动
from pymongo import MongoClient

def get_user(db, user_id: str):
    return db.users.find_one({"_id": user_id})
```

#### 5. HTTP 客户端

```python
# ✅ 正确：使用 httpx.AsyncClient
import httpx

async def call_api(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# ❌ 错误：使用 requests
import requests

def call_api(url: str):
    response = requests.get(url)
    return response.json()
```

#### 6. 并发处理

使用 `asyncio.gather()` 进行并发操作：

```python
# ✅ 正确：并发执行多个任务
async def fetch_multiple_users(user_ids: list[str]):
    tasks = [fetch_user_data(uid) for uid in user_ids]
    results = await asyncio.gather(*tasks)
    return results

# ❌ 错误：串行执行
async def fetch_multiple_users(user_ids: list[str]):
    results = []
    for uid in user_ids:
        result = await fetch_user_data(uid)
        results.append(result)
    return results
```

#### 7. 禁止在 for 循环中进行 I/O 操作 ⚠️

**💡 重要提示：避免在循环中进行串行 I/O 操作**

在 for 循环中进行数据库访问、API 调用等 I/O 操作会导致严重的性能问题，因为每次操作都需要等待前一次操作完成，无法充分利用异步并发的优势。

**❌ 错误示例：在循环中进行 I/O 操作**

```python
# 错误：在循环中串行访问数据库
async def get_users_info(user_ids: list[str]):
    results = []
    for user_id in user_ids:
        # 每次循环都要等待数据库返回，性能极差
        user = await db.users.find_one({"_id": user_id})
        results.append(user)
    return results

# 错误：在循环中串行调用 API
async def fetch_user_profiles(user_ids: list[str]):
    profiles = []
    for user_id in user_ids:
        # 每次循环都要等待 API 响应，浪费时间
        response = await api_client.get(f"/users/{user_id}")
        profiles.append(response.json())
    return profiles

# 错误：在循环中进行批量数据库插入
async def save_messages(messages: list[dict]):
    for msg in messages:
        # 每条消息单独插入，效率很低
        await db.messages.insert_one(msg)
```

**✅ 正确示例：使用并发或批量操作**

```python
# 正确：使用 asyncio.gather 并发执行
async def get_users_info(user_ids: list[str]):
    tasks = [db.users.find_one({"_id": uid}) for uid in user_ids]
    results = await asyncio.gather(*tasks)
    return results

# 正确：使用 asyncio.gather 并发调用 API
async def fetch_user_profiles(user_ids: list[str]):
    tasks = [api_client.get(f"/users/{uid}") for uid in user_ids]
    responses = await asyncio.gather(*tasks)
    return [r.json() for r in responses]

# 正确：使用批量插入操作
async def save_messages(messages: list[dict]):
    if messages:
        await db.messages.insert_many(messages)

# 正确：使用数据库的 in 查询替代循环查询
async def get_users_info(user_ids: list[str]):
    # 一次查询获取所有数据
    cursor = db.users.find({"_id": {"$in": user_ids}})
    results = await cursor.to_list(length=None)
    return results
```

**性能对比**

假设有 100 个用户，每次数据库查询耗时 10ms：
- ❌ 循环串行查询：100 × 10ms = 1000ms（1秒）
- ✅ 并发查询：~10ms（几乎同时完成）
- ✅ 批量查询：~10ms（单次查询）

**例外情况**

少数情况下可能需要在循环中进行 I/O，但必须满足以下条件：

1. **后续操作依赖前一次结果**：必须等待前一个操作完成才能进行下一个
2. **限流需求**：需要控制并发数量，避免对外部服务造成压力
3. **已获得开发负责人批准**

```python
# 允许：有依赖关系的串行操作（需注释说明原因）
async def process_workflow(steps: list[dict]):
    result = None
    for step in steps:
        # 每一步依赖前一步的结果，无法并发
        result = await execute_step(step, previous_result=result)
    return result

# 允许：使用信号量控制并发数（需注释说明原因）
async def fetch_with_rate_limit(urls: list[str]):
    # 限制最多 5 个并发请求，避免触发外部 API 限流
    semaphore = asyncio.Semaphore(5)
    
    async def fetch_one(url: str):
        async with semaphore:
            return await api_client.get(url)
    
    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks)
```

---

## 🕐 时区意识规范

### 核心原则

**💡 重要提示：所有时间字段必须具备时区意识**

在处理日期和时间数据时，必须确保所有时间字段都携带时区信息，避免因时区不明确导致的数据错误和业务问题。

**⚠️ 禁止直接使用 `datetime` 模块的标准方法**

项目统一使用 `common_utils/datetime_utils.py` 中的工具函数处理时间，禁止直接使用以下方法：
- ❌ `datetime.datetime.now()`
- ❌ `datetime.datetime.utcnow()`
- ❌ `datetime.datetime.today()`

必须使用项目提供的工具函数：
- ✅ `get_now_with_timezone()` - 获取当前时间（带时区）
- ✅ `from_timestamp()` - 从时间戳转换
- ✅ `from_iso_format()` - 从 ISO 格式字符串转换
- ✅ `to_iso_format()` - 转换为 ISO 格式字符串
- ✅ `to_timestamp()` / `to_timestamp_ms()` - 转换为时间戳

### 时区处理规则

#### 1. 输入数据时区要求

所有进入系统的时间字段都必须满足以下要求：

- **必须携带时区信息**：所有 datetime 类型的字段必须是 timezone-aware 的
- **默认时区**：如果输入数据不带时区信息，统一将其视为 **Asia/Shanghai（上海时区，UTC+8）**
- **存储格式**：在数据库中存储时建议统一转换为 UTC 时区，但必须保留时区信息

#### 2. Python 实现规范

**✅ 正确示例：使用项目工具函数**

```python
from common_utils.datetime_utils import (
    get_now_with_timezone,
    from_timestamp,
    from_iso_format,
    to_iso_format,
    to_timestamp_ms,
    to_timezone
)

# 方式1：获取当前时间（自动带上海时区）
now = get_now_with_timezone()
# 返回: datetime.datetime(2025, 9, 16, 20, 17, 41, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))

# 方式2：从时间戳转换（自动识别秒级/毫秒级，自动添加时区）
dt = from_timestamp(1758025061)
dt_ms = from_timestamp(1758025061000)

# 方式3：从 ISO 字符串转换（自动处理时区）
dt = from_iso_format("2025-09-15T13:11:15.588000")  # 无时区，自动添加上海时区
dt_with_tz = from_iso_format("2025-09-15T13:11:15+08:00")  # 有时区，保留原时区后转换

# 方式4：格式化为 ISO 字符串（自动包含时区）
iso_str = to_iso_format(now)
# 返回: "2025-09-16T20:20:06.517301+08:00"

# 方式5：转换为时间戳
ts = to_timestamp_ms(now)
# 返回: 1758025061123
```

**❌ 错误示例：直接使用 datetime 模块**

```python
import datetime

# ❌ 错误：禁止使用 datetime.datetime.now()
naive_dt = datetime.datetime.now()  # 时区不明确，禁止使用！

# ❌ 错误：禁止使用 datetime.datetime.utcnow()
dt = datetime.datetime.utcnow()  # Python 3.12+ 已废弃，禁止使用！

# ❌ 错误：禁止使用 datetime.datetime.today()
dt = datetime.datetime.today()  # 时区不明确，禁止使用！

# ❌ 错误：手动创建 naive datetime
naive_dt = datetime.datetime(2025, 1, 1, 12, 0, 0)  # 没有时区信息
```

**🔧 如何修复现有代码**

```python
# 旧代码（错误）
import datetime
now = datetime.datetime.now()

# 新代码（正确）
from common_utils.datetime_utils import get_now_with_timezone
now = get_now_with_timezone()

# ----------------

# 旧代码（错误）
from datetime import datetime
dt = datetime(2025, 1, 1, 12, 0, 0)

# 新代码（正确）
from common_utils.datetime_utils import from_iso_format
dt = from_iso_format("2025-01-01T12:00:00")  # 自动添加上海时区

# ----------------

# 旧代码（错误）
ts = int(datetime.now().timestamp() * 1000)

# 新代码（正确）
from common_utils.datetime_utils import get_now_with_timezone, to_timestamp_ms
ts = to_timestamp_ms(get_now_with_timezone())
```

#### 3. 时区转换示例

```python
from common_utils.datetime_utils import get_now_with_timezone, to_timezone
from zoneinfo import ZoneInfo

# 获取上海时间
dt_shanghai = get_now_with_timezone()

# 转换为 UTC
dt_utc = to_timezone(dt_shanghai, ZoneInfo("UTC"))

# 转换为其他时区
dt_ny = to_timezone(dt_shanghai, ZoneInfo("America/New_York"))
```

#### 4. 数据库操作规范

**MongoDB 示例**

```python
from common_utils.datetime_utils import get_now_with_timezone, from_iso_format

# ✅ 正确：插入带时区的时间
data = {
    "created_at": get_now_with_timezone(),
    "updated_at": get_now_with_timezone()
}
await collection.insert_one(data)

# ✅ 正确：查询时也使用带时区的时间
start_time = from_iso_format("2025-01-01T00:00:00")
results = await collection.find({"created_at": {"$gte": start_time}})
```

**PostgreSQL 示例**

```python
from common_utils.datetime_utils import get_now_with_timezone

# ✅ 正确：使用 timestamptz 类型
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

# Python 查询时
dt = get_now_with_timezone()
await conn.execute("INSERT INTO events (created_at) VALUES ($1)", dt)
```

#### 5. API 接口规范

**接收外部输入**

```python
from common_utils.datetime_utils import from_iso_format
import datetime

def process_datetime_input(dt_str: str) -> datetime.datetime:
    """处理外部输入的时间字符串"""
    try:
        # 使用工具函数解析，自动处理时区
        # 如果输入没有时区信息，自动添加上海时区
        dt = from_iso_format(dt_str)
        return dt
    except Exception as e:
        raise ValueError(f"Invalid datetime format: {dt_str}") from e
```

**返回输出**

```python
from common_utils.datetime_utils import to_iso_format
import datetime

# ✅ 正确：返回 ISO 8601 格式（包含时区）
def serialize_datetime(dt: datetime.datetime) -> str:
    """序列化 datetime 为 ISO 8601 格式"""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    # 使用工具函数格式化，自动包含时区信息
    return to_iso_format(dt)

# 示例输出："2025-01-01T12:00:00+08:00"
```

#### 6. 常见问题和注意事项

**Q: 为什么选择上海时区作为默认时区？**  
A: 项目主要服务中国用户，上海时区（Asia/Shanghai，UTC+8）是最常用的时区。

**Q: 可以使用 pytz 库吗？**  
A: Python 3.9+ 推荐使用标准库的 `zoneinfo`，它是官方推荐的时区处理方案，`pytz` 已逐步被淘汰。

**Q: 数据库中应该存储 UTC 还是本地时区？**  
A: 建议存储 UTC 时区，在展示时再转换为用户所需时区。这样可以避免夏令时等问题。

**Q: 如何处理历史数据中的 naive datetime？**  
A: 需要编写数据迁移脚本，为所有 naive datetime 添加上海时区信息。参考[数据迁移规范](#数据迁移与-schema-变更流程)。

### 检查清单

在代码审查时，请确认以下事项：

- [ ] **禁止直接使用 `datetime.datetime.now()`**，必须使用 `get_now_with_timezone()`
- [ ] **禁止直接使用 `datetime.datetime.utcnow()`** 或 `datetime.datetime.today()`
- [ ] 所有时间获取都通过 `common_utils/datetime_utils.py` 中的工具函数
- [ ] 从外部输入解析的时间使用 `from_iso_format()` 或 `from_timestamp()` 处理
- [ ] 时间格式化使用 `to_iso_format()` 而不是手动调用 `.isoformat()`
- [ ] 时间戳转换使用 `to_timestamp_ms()` 而不是手动计算
- [ ] 数据库 schema 使用了时区感知的类型（如 `timestamptz`）
- [ ] API 返回的时间字符串包含时区信息（ISO 8601 格式）
- [ ] 单元测试中使用的测试数据都带有时区信息

---

## 🏛️ 数据访问规范

### 核心原则

**💡 重要提示：所有外部存储访问必须通过 infra 层的 repository**

在处理数据库、搜索引擎等外部存储系统时，必须遵循严格的分层架构原则。所有数据读写操作都必须收敛到 `infra_layer` 的 `repository` 层，禁止在业务层或其他上层直接调用外部仓储能力。

**⚠️ 禁止在以下层级直接访问外部存储**
- ❌ `biz_layer`（业务层）
- ❌ `memory_layer`（记忆层）
- ❌ `agentic_layer`（Agent层）
- ❌ API 接口层（`api_specs`）
- ❌ 应用层（`app.py`、控制器等）

**✅ 必须通过以下方式访问**
- `infra_layer/adapters/out/persistence/repository/` - 数据库访问
- `infra_layer/adapters/out/search/repository/` - 搜索引擎访问

### 为什么需要这个规范？

#### 1. 职责分离

遵循六边形架构（Hexagonal Architecture）和整洁架构（Clean Architecture）原则：
- **业务层**：专注于业务逻辑，不关心数据来自何处
- **基础设施层**：负责所有外部系统的交互细节
- **隔离变化**：更换数据库或搜索引擎时，只需修改 infra 层

#### 2. 可测试性

```python
# ✅ 好处：业务层依赖抽象接口，易于 mock 测试
async def process_user_memory(user_id: str, memory_repo: MemoryRepository):
    """业务逻辑不依赖具体实现"""
    memories = await memory_repo.find_by_user_id(user_id)
    # 业务处理...
    
# 测试时可以轻松替换为 mock
mock_repo = MockMemoryRepository()
await process_user_memory("user_1", mock_repo)
```

#### 3. 代码复用与一致性

- 避免在多处重复编写相同的数据库查询逻辑
- 统一处理异常、日志、性能监控
- 统一处理数据转换、验证

#### 4. 性能优化集中管理

- 索引优化、查询优化在 repository 层统一实现
- 缓存策略统一管理
- 批量操作优化在一处完成，全项目受益

### 正确的架构分层

```
┌─────────────────────────────────────────┐
│  API Layer (api_specs, app.py)         │
│  - 接收请求，返回响应                      │
└─────────────┬───────────────────────────┘
              │ 调用
              ▼
┌─────────────────────────────────────────┐
│  Business Layer (biz_layer)             │
│  - 业务逻辑处理                           │
│  - 依赖抽象接口（Port）                    │
└─────────────┬───────────────────────────┘
              │ 依赖注入
              ▼
┌─────────────────────────────────────────┐
│  Memory Layer (memory_layer)            │
│  - 记忆管理逻辑                           │
│  - 依赖抽象接口（Port）                    │
└─────────────┬───────────────────────────┘
              │ 依赖注入
              ▼
┌─────────────────────────────────────────┐
│  Infrastructure Layer (infra_layer)     │
│  - Repository 实现（Adapter）             │
│  - 直接操作数据库/搜索引擎                  │
│  - MongoDB, PostgreSQL, ES, Milvus     │
└─────────────────────────────────────────┘
```

### 实现规范

#### ✅ 正确示例：通过 Repository 访问

**定义 Repository 接口（Port）**

```python
# core/ports/memory_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional

class MemoryRepository(ABC):
    """记忆仓储接口（抽象）"""
    
    @abstractmethod
    async def save(self, memory: Memory) -> str:
        """保存记忆"""
        pass
    
    @abstractmethod
    async def find_by_id(self, memory_id: str) -> Optional[Memory]:
        """根据ID查询记忆"""
        pass
    
    @abstractmethod
    async def find_by_user_id(self, user_id: str, limit: int = 100) -> List[Memory]:
        """根据用户ID查询记忆列表"""
        pass
    
    @abstractmethod
    async def search_semantic(self, query: str, user_id: str, top_k: int = 10) -> List[Memory]:
        """语义搜索"""
        pass
```

**实现 Repository（Adapter）**

```python
# infra_layer/adapters/out/persistence/repository/memory_mongo_repository.py
from motor.motor_asyncio import AsyncIOMotorDatabase
from core.ports.memory_repository import MemoryRepository
from core.domain.memory import Memory

class MemoryMongoRepository(MemoryRepository):
    """MongoDB 记忆仓储实现"""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self._collection = db["memories"]
    
    async def save(self, memory: Memory) -> str:
        result = await self._collection.insert_one(memory.to_dict())
        return str(result.inserted_id)
    
    async def find_by_id(self, memory_id: str) -> Optional[Memory]:
        doc = await self._collection.find_one({"_id": memory_id})
        return Memory.from_dict(doc) if doc else None
    
    async def find_by_user_id(self, user_id: str, limit: int = 100) -> List[Memory]:
        cursor = self._collection.find({"user_id": user_id}).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [Memory.from_dict(doc) for doc in docs]
    
    async def search_semantic(self, query: str, user_id: str, top_k: int = 10) -> List[Memory]:
        # 调用向量搜索（在 infra 层封装）
        # 这里可能还会调用 ElasticSearch 或 Milvus
        ...
```

**业务层使用 Repository**

```python
# biz_layer/services/memory_service.py
from core.ports.memory_repository import MemoryRepository
from core.domain.memory import Memory

class MemoryService:
    """记忆业务服务"""
    
    def __init__(self, memory_repo: MemoryRepository):
        # ✅ 依赖注入：依赖抽象接口，不依赖具体实现
        self._memory_repo = memory_repo
    
    async def create_memory(self, user_id: str, content: str) -> str:
        """创建记忆（业务逻辑）"""
        # 业务逻辑：构建领域对象
        memory = Memory(user_id=user_id, content=content)
        
        # ✅ 正确：通过 repository 保存
        memory_id = await self._memory_repo.save(memory)
        return memory_id
    
    async def get_user_memories(self, user_id: str) -> List[Memory]:
        """获取用户记忆列表"""
        # ✅ 正确：通过 repository 查询
        return await self._memory_repo.find_by_user_id(user_id)
    
    async def search_memories(self, user_id: str, query: str) -> List[Memory]:
        """搜索记忆"""
        # ✅ 正确：通过 repository 进行语义搜索
        return await self._memory_repo.search_semantic(query, user_id)
```

#### ❌ 错误示例：业务层直接访问数据库

```python
# ❌ 错误：在业务层直接使用 MongoDB 驱动
from motor.motor_asyncio import AsyncIOMotorClient

class MemoryService:
    def __init__(self, db_uri: str):
        # ❌ 业务层不应该直接连接数据库
        self._client = AsyncIOMotorClient(db_uri)
        self._db = self._client["memsys"]
    
    async def create_memory(self, user_id: str, content: str) -> str:
        # ❌ 业务层不应该直接操作 collection
        result = await self._db.memories.insert_one({
            "user_id": user_id,
            "content": content
        })
        return str(result.inserted_id)
```

```python
# ❌ 错误：在 memory_layer 直接使用 ElasticSearch
from elasticsearch import AsyncElasticsearch

class MemoryRetriever:
    def __init__(self, es_hosts: list):
        # ❌ 不应该在这一层直接创建 ES 客户端
        self._es = AsyncElasticsearch(hosts=es_hosts)
    
    async def search(self, query: str) -> list:
        # ❌ 不应该直接调用 ES API
        result = await self._es.search(index="memories", body={
            "query": {"match": {"content": query}}
        })
        return result["hits"]["hits"]
```

```python
# ❌ 错误：在 API 层直接访问数据库
from fastapi import APIRouter
from motor.motor_asyncio import AsyncIOMotorClient

router = APIRouter()
db_client = AsyncIOMotorClient("mongodb://localhost")

@router.get("/memories/{user_id}")
async def get_memories(user_id: str):
    # ❌ API 层不应该直接查询数据库
    db = db_client["memsys"]
    memories = await db.memories.find({"user_id": user_id}).to_list(100)
    return {"data": memories}
```

### 依赖注入配置

**使用依赖注入容器管理依赖关系**

```python
# application_startup.py 或 bootstrap.py
from dependency_injector import containers, providers
from infra_layer.adapters.out.persistence.repository.memory_mongo_repository import MemoryMongoRepository
from biz_layer.services.memory_service import MemoryService

class Container(containers.DeclarativeContainer):
    """依赖注入容器"""
    
    # 配置
    config = providers.Configuration()
    
    # 数据库连接
    mongodb_client = providers.Singleton(
        AsyncIOMotorClient,
        config.mongodb.uri
    )
    
    mongodb_database = providers.Singleton(
        lambda client: client[config.mongodb.database],
        client=mongodb_client
    )
    
    # Repository 层（基础设施）
    memory_repository = providers.Factory(
        MemoryMongoRepository,
        db=mongodb_database
    )
    
    # Service 层（业务逻辑）
    memory_service = providers.Factory(
        MemoryService,
        memory_repo=memory_repository
    )
```

### 搜索引擎访问规范

**ElasticSearch / Milvus 同样遵循 Repository 模式**

```python
# infra_layer/adapters/out/search/repository/semantic_memory_es_repository.py
from elasticsearch import AsyncElasticsearch
from typing import List

class SemanticMemoryESRepository:
    """ElasticSearch 语义记忆仓储"""
    
    def __init__(self, es_client: AsyncElasticsearch, index_name: str):
        self._es = es_client
        self._index = index_name
    
    async def index_memory(self, memory_id: str, content: str, embedding: List[float]):
        """索引记忆到 ES"""
        await self._es.index(
            index=self._index,
            id=memory_id,
            body={
                "content": content,
                "embedding": embedding
            }
        )
    
    async def search_by_vector(self, query_vector: List[float], top_k: int = 10) -> List[dict]:
        """向量搜索"""
        result = await self._es.search(
            index=self._index,
            body={
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                },
                "size": top_k
            }
        )
        return result["hits"]["hits"]
```

**业务层调用搜索 Repository**

```python
# memory_layer/retrievers/semantic_retriever.py
from infra_layer.adapters.out.search.repository.semantic_memory_es_repository import SemanticMemoryESRepository

class SemanticRetriever:
    """语义检索器（业务逻辑层）"""
    
    def __init__(self, search_repo: SemanticMemoryESRepository):
        # ✅ 依赖抽象，通过依赖注入获得 repository
        self._search_repo = search_repo
    
    async def retrieve_similar_memories(self, query_embedding: List[float], top_k: int = 10):
        """检索相似记忆"""
        # ✅ 通过 repository 访问搜索引擎
        results = await self._search_repo.search_by_vector(query_embedding, top_k)
        # 业务逻辑：过滤、排序、格式化等
        return self._process_results(results)
```

### 多数据源场景

**Repository 可以封装多个数据源的访问**

```python
# infra_layer/adapters/out/persistence/repository/memory_hybrid_repository.py
class MemoryHybridRepository(MemoryRepository):
    """混合记忆仓储：MongoDB + ElasticSearch"""
    
    def __init__(
        self,
        mongo_repo: MemoryMongoRepository,
        es_repo: SemanticMemoryESRepository
    ):
        self._mongo = mongo_repo
        self._es = es_repo
    
    async def save(self, memory: Memory) -> str:
        """保存到 MongoDB 和 ES"""
        # 保存到 MongoDB
        memory_id = await self._mongo.save(memory)
        
        # 同步到 ElasticSearch（异步任务或立即同步）
        await self._es.index_memory(
            memory_id=memory_id,
            content=memory.content,
            embedding=memory.embedding
        )
        
        return memory_id
    
    async def search_semantic(self, query: str, user_id: str, top_k: int = 10) -> List[Memory]:
        """语义搜索：ES 查询 + MongoDB 补充详情"""
        # 1. ES 搜索得到相关 ID
        es_results = await self._es.search_by_text(query, top_k)
        memory_ids = [hit["_id"] for hit in es_results]
        
        # 2. MongoDB 批量查询完整数据
        memories = await self._mongo.find_by_ids(memory_ids)
        return memories
```

### 检查清单

在编写或审查代码时，请确认以下事项：

- [ ] **数据库操作是否在 infra_layer/repository 中？**
- [ ] **搜索引擎操作是否在 infra_layer/repository 中？**
- [ ] **业务层是否依赖抽象接口（Port）而非具体实现？**
- [ ] **是否使用依赖注入传递 repository？**
- [ ] **是否避免在业务层/API层/应用层直接创建数据库连接？**
- [ ] **是否避免在业务层直接使用 MongoDB/PostgreSQL/ES/Milvus 的客户端？**
- [ ] **新增的 Repository 是否已注册到依赖注入容器？**
- [ ] **Repository 方法是否具有清晰的业务语义（而非暴露底层实现细节）？**

### 常见问题

**Q: 为什么不能在业务层直接用 MongoDB 驱动？**  
A: 违反了架构分层原则，导致业务逻辑与基础设施耦合，难以测试、难以替换数据源。

**Q: 简单的查询也要通过 Repository 吗？**  
A: 是的。即使是简单查询，也应该在 Repository 中封装。这样可以：
   - 统一管理所有数据访问
   - 后续优化时只需修改一处
   - 保持代码风格一致

**Q: Repository 方法应该返回 dict 还是领域对象？**  
A: 建议返回领域对象（如 `Memory`、`User`），这样业务层不需要关心数据的底层格式。

**Q: 如何处理复杂的联表查询？**  
A: 在 Repository 层封装复杂查询逻辑，对外提供语义化的方法。例如：
```python
async def find_memories_with_user_info(self, user_id: str) -> List[MemoryWithUser]:
    # Repository 内部处理联表或多次查询
    ...
```

**Q: 可以在 Repository 中调用其他 Repository 吗？**  
A: 可以，但要注意：
   - 避免循环依赖
   - 复杂的跨数据源逻辑建议放在业务层协调
   - Repository 职责应该单一

### 相关文档

- [六边形架构（Hexagonal Architecture）](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software))
- [整洁架构（Clean Architecture）](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [依赖注入模式](https://python-dependency-injector.ets-labs.org/)

---

## 📥 导入规范

### PYTHONPATH 管理

**💡 重要提示：PYTHONPATH 需要统一管理**

项目对 `PYTHONPATH` 和模块导入路径进行统一管理，涉及路径配置的变更建议与开发负责人沟通后统一配置。

#### 为什么需要统一管理？

- 导入路径混乱可能导致模块找不到或导入错误
- 不同环境（开发/测试/生产）路径不一致可能引发部署问题
- IDE 配置不统一可能影响团队协作
- 相对导入和绝对导入混用会增加代码维护难度

#### 管理范围

项目中以下目录的导入路径建议保持统一：

- `src/`：主要业务代码
- `tests/`：测试代码
- `unit_test/`：单元测试
- `evaluation/`：评估脚本
- 其他需要被导入的目录（如 `demo/` 等）

#### 推荐做法

1. **统一项目根目录**
   - 项目根目录为 `/Users/admin/memsys`（或部署环境对应路径）
   - src 目录已加入 PYTHONPATH，导入时直接从模块名开始

2. **导入规范示例**

```python
# ✅ 推荐：绝对导入（src 已在 PYTHONPATH 中）
from core.memory.manager import MemoryManager
from infra_layer.adapters.out.db import MongoDBAdapter
from tests.fixtures.mock_data import get_mock_user

# ✅ 推荐：测试文件中的导入
from unit_test.email_data_constructor import construct_email

# ❌ 不推荐：相对导入跨层级
from ...core.memory.manager import MemoryManager

# ❌ 不推荐：包含 src 前缀（src 已在 PYTHONPATH 中，无需前缀）
from src.core.memory.manager import MemoryManager

# ❌ 不推荐：使用 sys.path.append 临时修改路径
import sys
sys.path.append("../src")  # 可能造成环境不一致
```

3. **路径配置变更流程**

如果需要：
- 新增可导入的目录
- 修改现有目录的导入方式
- 调整 PYTHONPATH 配置

**建议按以下流程操作**：

1. **与开发负责人讨论**：说明变更原因和影响范围
2. **统一配置**：更新以下文件
   - `src/bootstrap.py`：启动入口
   - `auto.sh`：自动化脚本
   - `.vscode/settings.json` 或 `.idea` 配置
   - 部署脚本中的环境变量设置
3. **文档更新**：更新本文档和相关开发文档
4. **团队通知**：通知所有开发者同步配置

4. **IDE 配置（推荐统一）**

建议在 IDE 中将项目根目录标记为 Sources Root：

- **PyCharm**: 右键项目根目录 → Mark Directory as → Sources Root
- **VSCode**: 在 `.vscode/settings.json` 中配置：
  ```json
  {
    "python.analysis.extraPaths": [
      "${workspaceFolder}"
    ]
  }
  ```

### 优先使用绝对导入

**💡 重要提示：建议使用绝对导入，避免相对导入**

#### 为什么推荐绝对导入？

相对导入虽然在某些场景下更简洁，但存在以下问题：

- **可读性差**：`from ...core.memory import Manager` 不如 `from core.memory import Manager` 直观
- **重构困难**：移动文件时需要修改所有相对导入的层级
- **调试复杂**：堆栈追踪中相对导入路径不够清晰
- **工具支持**：IDE 和静态分析工具对绝对导入的支持更好
- **测试便利**：测试文件使用绝对导入更容易理解依赖关系

#### 导入方式对比

```python
# ✅ 推荐：绝对导入（src 已在 PYTHONPATH 中）
from core.memory.manager import MemoryManager
from core.memory.types import MemoryType, MemoryStatus
from infra_layer.adapters.out.db.mongodb import MongoDBAdapter
from common_utils.logger import get_logger

# ✅ 可接受：同一包内的相对导入（单层）
# 文件：src/core/memory/manager.py
from .types import MemoryType  # 同目录下
from .extractors.base import BaseExtractor  # 子目录

# ❌ 不推荐：跨层级的相对导入
from ...infra_layer.adapters import MongoDBAdapter
from ....common_utils.logger import get_logger

# ❌ 不推荐：向上多层的相对导入（难以维护）
from ......some_module import something
```

#### 使用规则

**推荐做法**：

1. **跨模块导入必须使用绝对导入**
   ```python
   # 从 src/core/memory/manager.py 导入到 src/biz_layer/service.py
   from core.memory.manager import MemoryManager  # ✅
   ```

2. **同一包内可以使用单层相对导入**
   ```python
   # 在 src/core/memory/manager.py 中
   from .types import MemoryType  # ✅ 同目录
   from .extractors.base import BaseExtractor  # ✅ 子目录
   ```

3. **避免向上多层的相对导入**
   ```python
   from ...infra_layer import something  # ❌ 应改为绝对导入
   from infra_layer import something  # ✅
   ```

#### 特殊场景说明

**场景 1：包内部的模块导入**

对于一个包内部的模块，如果需要相互导入：

```python
# 包结构：
# src/core/memory/
#   ├── __init__.py
#   ├── manager.py
#   ├── types.py
#   └── extractors/
#       ├── __init__.py
#       └── base.py

# 在 manager.py 中：
from .types import MemoryType  # ✅ 可接受的单层相对导入
from core.memory.types import MemoryType  # ✅ 也可以用绝对导入

# 在 extractors/base.py 中：
from ..types import MemoryType  # 🤔 可以，但绝对导入更好
from core.memory.types import MemoryType  # ✅ 推荐
```

**场景 2：测试文件的导入**

测试文件建议全部使用绝对导入：

```python
# tests/test_memory_manager.py
from core.memory.manager import MemoryManager  # ✅
from core.memory.types import MemoryType  # ✅
from tests.fixtures.mock_data import get_mock_data  # ✅
```

### __init__.py 使用规范

**💡 重要提示：不建议在 `__init__.py` 中编写任何代码**

#### 为什么要保持 `__init__.py` 为空？

- **导入副作用**：`__init__.py` 在包被导入时就会执行，任何代码都可能产生意外的副作用
- **循环依赖**：即使是简单的模块导出也容易导致循环导入问题
- **性能影响**：导入包时执行代码会影响启动性能和模块加载速度
- **可维护性**：代码散落在 `__init__.py` 中难以定位和维护
- **测试困难**：mock 和单元测试会变得复杂
- **隐式行为**：导入时的隐式执行增加了代码理解难度

#### 推荐用法

**✅ 推荐：保持空文件**

```python
# src/core/memory/__init__.py

# 空文件，仅作为 Python 包标识
# 不要在这里写任何代码
```

**如何导入模块？**

直接从具体的模块文件导入，不依赖 `__init__.py` 的 re-export：

```python
# ✅ 推荐：直接从模块文件导入
from core.memory.manager import MemoryManager
from core.memory.types import MemoryType, MemoryStatus
from core.memory.extractors.base import BaseExtractor

# ❌ 不推荐：依赖 __init__.py 的 re-export
from core.memory import MemoryManager  # 需要 __init__.py 中有导出代码
```

**❌ 不推荐的做法**

```python
# ❌ 不要在 __init__.py 中进行模块导出
# src/core/memory/__init__.py
from .manager import MemoryManager
from .types import MemoryType, MemoryStatus
from .extractors import BaseExtractor

__all__ = ["MemoryManager", "MemoryType", "MemoryStatus", "BaseExtractor"]
# 虽然看起来无害，但会增加循环依赖风险和维护成本
```

```python
# ❌ 不要在 __init__.py 中初始化全局对象
# src/core/memory/__init__.py
from .manager import MemoryManager

# 不要这样做！
global_memory_manager = MemoryManager()  # ❌
config = load_config()  # ❌
db_connection = connect_to_db()  # ❌
```

```python
# ❌ 不要在 __init__.py 中编写业务函数或类
# src/core/memory/__init__.py

def process_memory(data):  # ❌ 应该在单独的模块文件中
    # 业务逻辑...
    pass

class MemoryProcessor:  # ❌ 应该在单独的模块文件中
    pass
```

```python
# ❌ 不要在 __init__.py 中执行任何逻辑
# src/core/__init__.py

# 不要这样做！
import logging
logging.basicConfig(...)  # ❌ 副作用

if some_condition:  # ❌ 条件执行
    do_something()

for item in items:  # ❌ 循环逻辑
    process(item)

__version__ = "1.0.0"  # ❌ 即使是版本信息也不建议
```

#### 正确的代码组织方式

保持 `__init__.py` 为空，业务代码都放在独立的模块文件中：

```python
# 包结构：
# src/core/memory/
#   ├── __init__.py          # 空文件
#   ├── manager.py           # MemoryManager 类
#   ├── types.py             # 类型定义
#   ├── processor.py         # 业务处理函数
#   └── config.py            # 配置管理

# src/core/memory/__init__.py - 保持空文件
# (空文件，不写任何代码)

# src/core/memory/manager.py - 实际的业务逻辑
class MemoryManager:
    def __init__(self):
        # 实现细节...
        pass

# src/core/memory/processor.py - 业务函数
def process_memory(data):
    # 实际的业务逻辑...
    pass

# 使用时直接从具体模块导入
from core.memory.manager import MemoryManager
from core.memory.types import MemoryType, MemoryStatus
from core.memory.processor import process_memory
```

#### 检查清单

在编写或审查 `__init__.py` 时，请确认：

- [ ] 文件是否为空（或仅包含注释）？
- [ ] 是否没有任何 import 语句？
- [ ] 是否没有定义任何变量、常量？
- [ ] 是否没有创建全局对象实例？
- [ ] 是否没有定义任何类或函数？
- [ ] 是否没有执行任何逻辑？

**如果以上任何一项回答"否"，请将代码移到独立的模块文件中。**

---

## 🌿 分支管理规范

### 分支类型说明

| 分支 | 描述 | 备注 |
|------|------|------|
| `master` | 稳定版本；仅限 bug 修复切出，`release/xxx` 和 `hotfix/xxx` 合并进来 | 生产环境部署分支 |
| `dev` | 日常开发版本；持续提交代码 | 如果已经开始定版 & 提交是本次版本，往 `release` 中提交；不紧急的小 bug & 功能合并 `dev`，搭下次发布的车 |
| `release/YYMMDD` | 定版分支；先发布测试，然后发布正式；首先 `dev` 合并 `master`，然后从 `dev` 切出；真正发布后合并回 `master`、`dev` | 目前不定期（群内通知）；仅限本次发布的 bug 提交 or 代码提交 |
| `feature/xxxx` | 单个周期内、小功能；合并到 `dev` 或者某个 `release` | 合并到 `dev` 可直接合并；合并到 `release` 建议 MR |
| `bugfix/xxxx` | 单个周期内、小 bug；合并到 `dev` 或者某个 `release` | 合并到 `dev` 可直接合并；合并到 `release` 建议 MR |
| `long/xxx` | 跨周期、大功能；`dev` 切出，合并到 `dev` 或者某个 `release` | 单独测试额外开新测试环境（端口/地址区分）；日常合并 `dev`，避免最后冲突太多；建议 MR |
| `hotfix/xxxx` | bug 修复；`master` 切出来，MR 到 `master` 分支（`dev` 如果有需要） | 发布之后才有这个分支；正常开发阶段的 bug 直接 `dev` 上合并，定版但还没发布的时候合并到 `release`，紧急的、当前没定版的才用这个 bug 分支；建议 MR |

### 环境与分支对应关系

| 环境 | 可能分支 | 备注 |
|------|----------|------|
| 正式 | `master` 分支 | 稳定版本 |
|      | `release/xxx` 分支 | 定版发布之后并且 bug 修复之前 |
| 测试 | `dev` 分支 | 日常开发阶段 |
|      | `release/xxx` 分支 | 定版测试阶段 |
|      | `hotfix/xxxx` | 紧急 bug 修复 |

### 版本标签规范

| Tag | 描述 | 备注 |
|-----|------|------|
| `X.Y.Z` | 版本号：大版本.迭代版本.Bug修复 | 跟迭代不一定同步，需要的时候加 |

- **X (大版本号)**：重大架构变更或不兼容更新
- **Y (迭代版本号)**：功能迭代、新特性添加
- **Z (修复版本号)**：Bug 修复、小优化

### 分支操作流程

#### 1. 日常开发（feature/bugfix）

```bash
# 从 dev 创建功能分支
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name

# 开发完成后
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name

# 合并到 dev（小功能可直接合并）
git checkout dev
git merge feature/your-feature-name
git push origin dev

# 删除功能分支
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

#### 2. 发版流程（release）

```bash
# 1. 先让 dev 合并 master（确保包含最新的 hotfix）
git checkout dev
git pull origin dev
git merge master
git push origin dev

# 2. 从 dev 创建 release 分支
git checkout -b release/$(date +%y%m%d)
git push origin release/$(date +%y%m%d)

# 3. 测试阶段的 bug 修复
git checkout release/$(date +%y%m%d)
# ... 修复 bug ...
git commit -m "fix: bug description"
git push origin release/$(date +%y%m%d)

# 4. 发布后合并回 master 和 dev
git checkout master
git merge release/$(date +%y%m%d)
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin master --tags

git checkout dev
git merge release/$(date +%y%m%d)
git push origin dev
```

#### 3. 紧急修复（hotfix）

```bash
# 从 master 创建 hotfix 分支
git checkout master
git pull origin master
git checkout -b hotfix/critical-bug-fix

# 修复完成后，建议走 MR 流程
git add .
git commit -m "hotfix: critical bug description"
git push origin hotfix/critical-bug-fix

# 创建 Merge Request 到 master
# 合并后记得同步到 dev
git checkout dev
git merge master
git push origin dev
```

#### 4. 长期功能开发（long）

```bash
# 从 dev 创建长期分支
git checkout dev
git pull origin dev
git checkout -b long/big-feature

# 定期合并 dev，避免冲突积累
git checkout long/big-feature
git merge dev

# 功能完成后，建议走 MR 流程合并到 dev 或 release
```

### 分支合并统一处理规范

**⚠️ 重要提示：以下分支合并操作需要由开发或运维负责人统一处理**

为了保证代码质量和发布流程的规范性，以下几类分支合并操作需要由开发负责人或运维负责人统一管理和执行：

#### 需要统一处理的合并操作

1. **长期功能分支合并到 dev**
   - `long/xxx` → `dev`
   - 原因：长期功能分支通常涉及大量代码变更，需要评估影响范围和潜在冲突

2. **dev 切出 release 分支**
   - `dev` → `release/YYMMDD`
   - 原因：发版节点需要统一协调，确保版本内容完整且符合发版要求

3. **release 合并回 dev**
   - `release/YYMMDD` → `dev`
   - 原因：确保发布分支的 bug 修复能正确同步回主开发分支

#### 操作流程

**开发者操作**：
```bash
# 1. 确保分支代码已推送到远程
git push origin your-branch

# 2. 联系开发负责人或运维负责人
# 3. 说明合并需求：
#    - 源分支名称
#    - 目标分支名称
#    - 合并原因和影响范围
#    - 是否已完成测试
```

**负责人操作**：
```bash
# 1. 检查代码质量和测试情况
# 2. 评估冲突和影响范围
# 3. 选择合适的时间窗口执行合并
# 4. 合并完成后通知相关人员
```

#### 为什么需要统一处理？

- **代码质量把控**：确保合并的代码经过充分测试和审查
- **版本管理规范**：避免版本混乱和发布流程不规范
- **冲突处理专业**：复杂冲突需要有经验的人员处理
- **团队协调统一**：避免多人同时操作导致的混乱
- **风险控制**：重要分支的合并需要有回滚预案

#### 注意事项

- 小功能分支（`feature/xxx`、`bugfix/xxx`）合并到 `dev` 可由开发者自行操作
- 紧急 `hotfix` 合并到 `master` 建议走 MR 流程并由负责人审查
- 所有涉及 `release` 和 `master` 的合并都建议经过负责人确认

---

## 🔍 Code Review 流程

### Code Review 推荐场景

以下情况建议经过 Code Review（创建 Merge Request/Pull Request）：

#### 1. 数据相关变更

- 💾 **数据库 Migration 脚本**
  - `migrations/` 目录下的新增或修改
  - 数据结构变更（字段增删改、索引变更）
  
- 💾 **数据修复脚本**
  - `devops_scripts/data_fix/` 目录下的脚本
  - 涉及批量数据修改的脚本
  - ElasticSearch 数据同步脚本

#### 2. 依赖包变更

- 📦 **依赖包的增加**
  - 新增生产依赖或开发依赖
  - 说明添加原因和用途
  
- 📦 **依赖包的删除**
  - 确认无其他模块依赖
  - 说明删除原因
  
- 📦 **依赖包的版本升级**
  - 重大版本升级（如 1.x → 2.x）
  - 涉及 API 变更的升级
  - 提供升级影响评估

相关变更会影响：
- `pyproject.toml`
- `uv.lock`

#### 3. 基础设施变更

- 🏗️ **基建新增或变更**
  - 新增中间件、数据库连接池等基础组件
  - 修改基础设施配置（数据库、Redis、Kafka 等）
  - 新增或修改 `infra_layer/` 下的适配器
  
- 🏗️ **应用启动加载流程**
  - 修改 `bootstrap.py`
  - 修改 `application_startup.py`
  - 修改 `base_app.py`
  - 修改依赖注入容器配置

#### 4. 重要分支合并

- 🔀 合并到 `release/xxx` 分支
- 🔀 合并到 `master` 分支（hotfix）
- 🔀 `long/xxx` 长期分支合并到 `dev`

### 数据迁移与 Schema 变更流程

**⚠️ 重要原则：提前规划，充分沟通**

上线新功能涉及数据修复或 Schema 迁移时，应尽早与研发负责人、运维负责人讨论，确保数据迁移方案的可行性和后续实施的时间安排。

#### 为什么需要提前沟通？

数据迁移和 Schema 变更是高风险操作，可能影响：

- **数据完整性**：数据结构变更可能导致数据丢失或损坏
- **服务可用性**：大规模数据迁移可能影响服务性能
- **回滚复杂度**：Schema 变更后的回滚往往比代码回滚更复杂
- **时间窗口**：需要预留足够的时间进行数据迁移和验证
- **多团队协作**：涉及研发、测试、运维多个团队的配合

#### 数据迁移流程建议

##### 1. 方案设计阶段

**时间点**：功能开发前或开发初期

**操作步骤**：
- 与研发负责人讨论：
  - 数据结构变更的必要性
  - 迁移方案的技术可行性
  - 是否需要兼容旧数据格式
  - 预估数据量和迁移时长
  
- 与运维负责人讨论：
  - 迁移对服务性能的影响
  - 是否需要停机维护
  - 回滚方案和应急预案
  - 数据备份策略

**输出物**：
- 数据迁移方案文档
- 风险评估报告
- 时间排期计划

##### 2. 脚本开发阶段

**时间点**：功能开发中期

**操作步骤**：
- 编写 Migration 脚本或数据修复脚本
- 在开发环境测试脚本
- 评估脚本性能（处理速度、资源占用）
- 准备数据验证脚本
- 编写回滚脚本

**注意事项**：
- 脚本必须经过 Code Review（见上文"数据相关变更"）
- 脚本应支持增量执行和断点续传
- 添加详细的日志记录
- 考虑批量处理，避免一次性加载大量数据

##### 3. 测试验证阶段

**时间点**：功能开发后期

**操作步骤**：
- 在测试环境执行完整的迁移流程
- 验证数据完整性和正确性
- 测试新功能与迁移后数据的兼容性
- 测试回滚流程的有效性
- 记录迁移耗时和资源占用

**验证清单**：
- [ ] 数据量统计正确（迁移前后对比）
- [ ] 数据格式符合预期
- [ ] 索引正确建立
- [ ] 新功能运行正常
- [ ] 旧数据兼容性验证通过
- [ ] 回滚脚本测试成功

##### 4. 上线实施阶段

**时间点**：发布日前与运维确认

**操作步骤**：

1. **与运维确认实施时间**
   - 确定迁移窗口（避开业务高峰）
   - 确认是否需要停机维护
   - 通知相关方（产品、客服等）

2. **准备工作**
   - 备份生产数据
   - 准备监控脚本
   - 准备回滚脚本
   - 准备应急预案

3. **执行迁移**
   - 按照既定方案执行
   - 实时监控迁移进度
   - 记录详细日志

4. **验证与监控**
   - 验证数据迁移结果
   - 监控服务运行状态
   - 观察性能指标
   - 收集用户反馈

##### 5. 后续跟踪

**时间点**：上线后 1-3 天

**操作步骤**：
- 持续监控数据和服务状态
- 处理遗留问题
- 总结迁移经验
- 更新文档

#### 常见场景示例

| 场景 | 提前沟通时间 | 关键讨论点 |
|------|--------------|------------|
| **新增字段** | 开发初期（1-2 周前） | 默认值策略、索引建立、是否需要回填历史数据 |
| **字段类型变更** | 方案设计阶段（2-3 周前） | 数据转换规则、不兼容数据处理、回滚方案 |
| **大规模数据修复** | 方案设计阶段（2-4 周前） | 数据量评估、迁移时长、分批策略、停机计划 |
| **索引重建** | 方案设计阶段（1-2 周前） | 对性能的影响、执行时间窗口、在线/离线方式 |
| **数据归档/清理** | 方案设计阶段（2-3 周前） | 归档策略、数据备份、恢复机制 |

#### 相关文档

- [MongoDB 迁移指南](./mongodb_migration_guide.md)
- 数据修复脚本目录：`src/devops_scripts/data_fix/`
- Migration 脚本目录：`src/migrations/`

### Code Review 流程

#### 提交者建议

1. **创建 Merge Request**
   - 填写清晰的标题和描述
   - 说明变更原因和影响范围
   - 关联相关 Issue 或需求

2. **自查清单**
   - [ ] 代码通过 pre-commit 检查
   - [ ] 相关单元测试已添加/更新
   - [ ] 文档已更新（如有必要）
   - [ ] 无明显的性能问题
   - [ ] 无安全隐患

3. **指定审查者**
   - 在 Merge Request 中指定审查者（见文档末尾联系人）
   - 明确标注变更类型：
     - 📊 数据变更（Migration/数据修复脚本）
     - 📦 依赖变更（pyproject.toml/uv.lock）
     - 🏗️ 基建变更（infra_layer/启动流程）
     - 🔀 重要分支合并（release/master/long）

#### 审查者工作

1. **代码质量审查**
   - 代码逻辑正确性
   - 代码可读性和可维护性
   - 是否符合项目规范

2. **风险评估**
   - 数据安全风险（特别关注数据脚本）
   - 性能影响（异步代码、数据库查询）
   - 兼容性问题（依赖升级、API 变更）

3. **审查反馈**
   - 提出明确的修改建议
   - 标注严重程度（Must Fix / Should Fix / Nice to Have）
   - 及时响应（尽量 24 小时内）

### MR 描述模板

```markdown
## 变更类型
- [ ] Feature (新功能)
- [ ] Bugfix (Bug 修复)
- [ ] Hotfix (紧急修复)
- [ ] Refactor (重构)
- [ ] 📊 Migration (数据迁移) - 建议 Code Review
- [ ] 📦 Dependency (依赖变更) - 建议 Code Review
- [ ] 🏗️ Infrastructure (基建变更) - 建议 Code Review

## 审查者
<!-- 指定审查者，见文档末尾联系人信息 -->

## 变更说明
<!-- 简要描述本次变更的内容和原因 -->

## 影响范围
<!-- 说明受影响的模块、服务或功能 -->

## 测试情况
- [ ] 代码通过 pre-commit 检查
- [ ] 单元测试已通过
- [ ] 集成测试已通过
- [ ] 手动测试已完成

## 风险评估
<!-- 如涉及数据/依赖/基建变更，说明风险和回滚方案 -->

## 相关文档
<!-- 关联的需求文档、设计文档或 Issue -->

## 截图/日志
<!-- 如有必要，提供截图或关键日志 -->
```

### 哪些变更建议 Code Review？

以下情况建议创建 MR 并指定审查者：

- 💾 **数据相关变更**（Migration、数据修复脚本）
- 📦 **依赖包的增删改**（pyproject.toml、uv.lock）
- 🏗️ **基础设施变更**（infra_layer、启动流程）
- 🔀 **合并到 release/xxx 或 master 分支**
- 🔀 **long/xxx 长期分支合并到 dev**

**小功能/小 bug 合并到 dev** 可视情况决定是否需要 MR。

### 如何提交 Merge Request？

```bash
# 1. 推送你的分支到远程
git push origin your-branch-name

# 2. 在 Git 平台（GitLab/GitHub/Gitee）创建 MR/PR
#    - 源分支：your-branch-name
#    - 目标分支：dev/release/master
#    - 审查者：见文档末尾联系人
#    - 使用上面的 MR 描述模板填写详细信息

# 3. 等待审查，根据反馈意见修改代码
git add .
git commit -m "fix: 根据 code review 意见修改 xxx"
git push origin your-branch-name

# 4. 审查通过后合并
#    - 由审查者确认后合并
#    - 或自行合并后通知相关人员
```

---

## 📚 相关文档

- [项目入门指南](./getting_started.md)
- [开发指南](./development_guide.md)
- [依赖管理指南](./project_deps_manage.md)
- [Bootstrap 使用说明](./bootstrap_usage.md)
- [MongoDB 迁移指南](./mongodb_migration_guide.md)

---

## ❓ 常见问题

### Q1: 忘记安装 pre-commit hook 怎么办？

```bash
pre-commit install
pre-commit run --all-files  # 对现有代码运行检查
```

### Q2: 不小心用 pip 安装了包怎么办？

```bash
# 1. 卸载用 pip 安装的包
pip uninstall <package-name>

# 2. 重新用 uv 安装
uv add <package-name>

# 3. 重新同步环境
uv sync
```

### Q3: 分支合并冲突怎么办？

```bash
# 1. 确保本地分支是最新的
git checkout your-branch
git pull origin your-branch

# 2. 合并目标分支
git merge target-branch

# 3. 解决冲突后提交
git add .
git commit -m "merge: resolve conflicts with target-branch"
```

### Q4: 如何在异步代码中调用同步库？

```python
import asyncio

async def use_sync_library():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # 使用默认线程池
        sync_function,
        arg1,
        arg2
    )
    return result
```

---

## 👤 联系人

### 开发负责人

如有以下事项，建议与开发负责人沟通：

- 线程/进程使用方案讨论
- PYTHONPATH 路径配置变更
- Code Review 审查请求
- 数据脚本、依赖变更、基建改动的技术方案

**当前负责人**：zhanghui

### 运维负责人

如有以下事项，请联系运维负责人：

- 开发环境配置获取（数据库、中间件连接信息）
- 服务访问权限申请
- 环境配置问题排查
- 新增配置项或环境需求
- 网络连接、VPN 等基础设施问题

**当前负责人**：jianhua

---

**最后更新时间**: 2025-10-31

