# Development Standards

This document introduces various standards and best practices in the project development process to help the team maintain code quality and collaboration efficiency.

---

## 🚀 TL;DR (Core Principles)

### Quick Start for Newcomers (3 Steps)
```bash
uv sync --group dev-full    # Sync dependencies
pre-commit install          # Install code check hooks
```

### Core Conventions

**📦 Dependency Management**  
Use `uv add/remove` to manage dependencies, avoid direct `pip install` to maintain consistency of dependency lock files

**🎨 Code Style**  
Pre-commit checks run automatically on commit (black/ruff/isort) to keep code style consistent

**⚡️ Full Async Architecture**  
Single Event Loop, use `async/await` for I/O operations, discuss with development lead before using threads/processes

**🚫 No I/O in Loops**  
Prohibit database access and API calls in for loops, use batch operations instead

**🕐 Timezone Awareness**  
All time fields must carry timezone information. Input without timezone is treated as Shanghai timezone (Asia/Shanghai, UTC+8). Do not use `datetime.datetime.now()`, must use utility functions from `common_utils/datetime_utils.py`

**📥 Import Standards**  
- PYTHONPATH management: Project module import starting paths (src/tests/demo etc.) need unified management, communicate with development lead before changes
- Prefer absolute imports (e.g. `from core.memory import MemoryManager`), avoid relative imports (e.g. `from ...core import`)

**📝 __init__.py Standards**  
Not recommended to write any code in `__init__.py`, keep it empty

**🌿 Branch Standards**  
`dev` for daily development, `release/YYMMDD` for versioned releases, `long/xxx` for long-term feature development, `hotfix` for emergency fixes

**🔀 Unified Branch Merge Handling**  
Merging `long/xxx` to `dev`, cutting `release` from `dev`, merging `release` back to `dev` needs to be handled uniformly by development or operations lead

**🔍 Code Review Requirements**  
Adding data migration scripts, adding/upgrading/downgrading third-party dependencies, infrastructure framework code changes, merging release branches must go through MR process

**💾 Data Migration Standards**  
For new features involving data fixes or Schema migration, discuss feasibility and implementation timing with development and operations as early as possible

**🏛️ Data Access Standards**  
All database, search engine and other external storage read/write operations must be converged to infra layer repository methods. Direct calls to external repositories in upper layers are prohibited

### 📖 Quick Navigation

- Don't know how to install dependencies? → [Dependency Management Standards](#dependency-management-standards)
- Need database/middleware configuration? → [Development Environment Configuration Standards](#development-environment-configuration-standards)
- Always getting errors before commit? → [Code Style Standards](#code-style-standards)
- Not sure if you can use threads? → [Async Programming Standards](#async-programming-standards)
- Can I do database queries in loops? → [Prohibit I/O Operations in for Loops](#7-prohibit-io-operations-in-for-loops-)
- How to handle time fields? → [Timezone Awareness Standards](#timezone-awareness-standards)
- Where should database queries be written? → [Data Access Standards](#data-access-standards)
- Import path errors? → [Import Standards](#import-standards)
- Don't know which branch to use? → [Branch Management Standards](#branch-management-standards)
- Need to submit MR? → [Code Review Process](#code-review-process)
- Need data migration? → [Data Migration and Schema Change Process](#data-migration-and-schema-change-process)

---

## 📋 Table of Contents

- [TL;DR (Quick Start)](#tldr-quick-start)
- [Dependency Management Standards](#dependency-management-standards)
- [Development Environment Configuration Standards](#development-environment-configuration-standards)
- [Code Style Standards](#code-style-standards)
- [Async Programming Standards](#async-programming-standards)
- [Timezone Awareness Standards](#timezone-awareness-standards)
- [Data Access Standards](#data-access-standards)
- [Import Standards](#import-standards)
  - [PYTHONPATH Management](#pythonpath-management)
  - [Prefer Absolute Imports](#prefer-absolute-imports)
  - [__init__.py Usage Standards](#__init__py-usage-standards)
- [Branch Management Standards](#branch-management-standards)
- [Code Review Process](#code-review-process)
  - [Data Migration and Schema Change Process](#data-migration-and-schema-change-process)

---

## 📦 Dependency Management Standards

### Using uv for Dependency Management

**💡 Important Note: Recommended to use uv for dependency management**

The project uses `uv` as the dependency management tool. It's recommended to avoid using `pip install` directly to install packages for the following reasons:

- Dependency versions may be inconsistent
- `uv.lock` file cannot be automatically updated
- Team member environments may differ
- May affect production environment deployment

### Correct Operations

#### 1. Install/Sync Dependencies

```bash
# Sync all dependencies (first install or after updates)
uv sync --group dev-full

```

#### 2. Add New Dependencies

```bash
# Add production dependency
uv add <package-name>

# Add development dependency
uv add --dev <package-name>

# Specify version
uv add <package-name>==<version>
```

#### 3. Remove Dependencies

```bash
uv remove <package-name>
```

#### 4. Update Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update specific dependency
uv add <package-name> --upgrade
```

### Related Documentation

For detailed dependency management guide, please refer to: [project_deps_manage.md](./project_deps_manage.md)

---

## 🔧 Development Environment Configuration Standards

### Environment Configuration Description

The project depends on various databases and middleware. To ensure consistency and security of the development environment, these configurations are uniformly managed and distributed by the operations team.

#### Configuration Items Involved

Development environment typically requires the following configurations:

**Database Configuration**
- MongoDB connection information
- PostgreSQL connection information
- Redis connection information

**Middleware Configuration**
- Kafka connection configuration
- ElasticSearch connection configuration
- Other message queues or cache services

**Third-party Service Configuration**
- API keys and access credentials
- Object storage configuration
- Other external service credentials

### How to Obtain Configuration

#### 1. New Team Members

For developers newly joining the project, please follow this process to obtain configuration:

1. **Contact Operations Lead** (see contact information at end of document)
2. **Explain Requirements**:
   - Your name and role
   - Required environment (development environment/test environment)
   - Specific services you need access to
3. **Receive Configuration**: Operations lead will provide configuration files or environment variables
4. **Local Configuration**: Put configuration information in the project's `config.json` or `.env` file (note: these files are in `.gitignore` and won't be committed to the repository)

#### 2. Configuration File Locations

```bash
# Configuration files in project root (do not commit to git)
config.json          # Main configuration file
.env                 # Environment variable configuration
env.template         # Configuration template (can refer to but need to fill in real values)
```

#### 3. Environment Variable Examples

Refer to the `env.template` file, your `.env` file typically contains the following types of configuration:

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

### Configuration Management Precautions

#### ⚠️ Security Standards

1. **Prohibit Committing Sensitive Configuration**
   - All configuration files containing passwords, keys, tokens must not be committed to git
   - Check if `.gitignore` includes configuration files before committing
   - Using pre-commit hook can help detect sensitive information

2. **Configuration File Permissions**
   - Local configuration files should be set with appropriate permissions (readable only by current user)
   - Do not paste configuration content directly in public places (such as chat logs, documents)

3. **Configuration Update Notification**
   - If configuration is updated, operations team will notify relevant developers
   - Update local configuration promptly after receiving notification

#### 🔄 Configuration Change Process

If you need to:
- Add new configuration items
- Modify configuration structure
- Add new environments or services

**Recommended process**:

1. **Discuss with Development Lead**: Confirm necessity and impact scope of configuration changes
2. **Contact Operations Lead**: Explain configuration requirements and reasons for change
3. **Update Configuration Template**: Update `env.template` and related documentation
4. **Team Notification**: Notify all developers to sync and update local configuration

#### 📝 Configuration Problem Troubleshooting

**Common Issues**:

1. **Connection Failed**
   - Check network connection (are you on company network or VPN)
   - Confirm if configuration information is correct
   - Contact operations lead to confirm service status

2. **Insufficient Permissions**
   - Confirm if account has been authorized
   - Contact operations lead to apply for appropriate permissions

3. **Configuration Expired**
   - Regularly check if configuration needs updating
   - Pay attention to configuration change information in team notifications

### Different Environment Descriptions

| Environment | Purpose | Configuration Source | Notes |
|------|------|----------|------|
| **Development Environment** | Local development and debugging | Provided by operations | Usually connects to development database, data can be tested freely |
| **Test Environment** | Integration testing and functional testing | Automatic deployment configuration | Connects to test database, data reset regularly |
| **Production Environment** | Officially running service | Strictly controlled by operations | Only operations and authorized personnel can access |

**Note**: Developers typically only need development environment configuration. Test environment and production environment configurations are managed by CI/CD and operations team.

---

## 🎨 Code Style Standards

### Pre-commit Hook Configuration

The project uses `pre-commit` to unify code style. It's recommended to install the pre-commit hook after cloning the project for the first time.

#### Installation Steps

```bash
# 1. Ensure development dependencies are synced
uv sync --dev

# 2. Install pre-commit hook
pre-commit install
```

#### Purpose

Pre-commit hook automatically executes the following checks before each commit:

- **Code Formatting**: Use black/ruff to format Python code
- **Import Sorting**: Use isort to sort import statements
- **Code Checking**: Use ruff/flake8 for code quality checks
- **Type Checking**: Use pyright/mypy for type checking
- **YAML/JSON Format**: Check configuration file format
- **Trailing Whitespace**: Remove trailing spaces at end of files

#### Manual Checks

```bash
# Run checks on all files
pre-commit run --all-files

# Run checks on staged files
pre-commit run
```

---

## ⚡️ Async Programming Standards

### Full Async Architecture Principles

The project adopts **full async architecture**, based on the following principles:

#### 1. Single Event Loop Principle

- **The entire application uses one main Event Loop**
- Avoid creating new Event Loop in code (`asyncio.new_event_loop()`)
- Avoid using `asyncio.run()` to start new loop in async context

#### 2. About Using Threads and Processes ⚠️

**💡 Important Note: Use multithreading and multiprocessing carefully**

The project is based on single Event Loop full async architecture. It's recommended to avoid the following operations:

```python
# ❌ Not recommended: Creating threads
import threading
thread = threading.Thread(target=some_function)
thread.start()

# ❌ Not recommended: Using thread pool (unless special circumstances)
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

# ❌ Not recommended: Creating processes
import multiprocessing
process = multiprocessing.Process(target=some_function)
process.start()

# ❌ Not recommended: Using process pool
from concurrent.futures import ProcessPoolExecutor
executor = ProcessPoolExecutor()
```

**Why not recommended?**
- May break single Event Loop architecture and cause concurrency issues
- Thread safety issues are complex and prone to race conditions
- Resource management is more difficult and may cause resource leaks
- May affect normal working of async context (contextvars)
- Debugging difficulty increases, stack tracing is more complex

**Special Scenario Handling**

If you really need to use threads or processes (e.g., CPU-intensive computing, calling third-party libraries that don't support async), it's recommended to:

1. **Discuss solution with development lead in advance**
2. Explain why async solution cannot meet requirements
3. Provide resource management solution (ensure threads/processes are properly closed)
4. Go through Code Review

**Few allowed scenario examples**:

```python
# ✅ Special scenario: Calling sync library that doesn't support async (after discussion)
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Globally shared thread pool, limit max threads
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

async def call_sync_library(data):
    """Call third-party library that doesn't support async (confirmed with lead)"""
    loop = asyncio.get_event_loop()
    # Run in thread pool to avoid blocking main loop
    result = await loop.run_in_executor(
        _EXECUTOR,
        sync_blocking_function,
        data
    )
    return result
```

#### 3. Async Function Definition

I/O operations should use async functions:

```python
# ✅ Correct: Async function
async def fetch_user_data(user_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/users/{user_id}")
        return response.json()

# ❌ Wrong: Sync I/O
def fetch_user_data(user_id: str) -> dict:
    response = requests.get(f"/users/{user_id}")
    return response.json()
```

#### 4. Database Operations

```python
# ✅ Correct: Using async database driver
from motor.motor_asyncio import AsyncIOMotorClient

async def get_user(db, user_id: str):
    return await db.users.find_one({"_id": user_id})

# ❌ Wrong: Using sync driver
from pymongo import MongoClient

def get_user(db, user_id: str):
    return db.users.find_one({"_id": user_id})
```

#### 5. HTTP Client

```python
# ✅ Correct: Using httpx.AsyncClient
import httpx

async def call_api(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# ❌ Wrong: Using requests
import requests

def call_api(url: str):
    response = requests.get(url)
    return response.json()
```

#### 6. Concurrent Processing

Use `asyncio.gather()` for concurrent operations:

```python
# ✅ Correct: Execute multiple tasks concurrently
async def fetch_multiple_users(user_ids: list[str]):
    tasks = [fetch_user_data(uid) for uid in user_ids]
    results = await asyncio.gather(*tasks)
    return results

# ❌ Wrong: Serial execution
async def fetch_multiple_users(user_ids: list[str]):
    results = []
    for uid in user_ids:
        result = await fetch_user_data(uid)
        results.append(result)
    return results
```

#### 7. Prohibit I/O Operations in for Loops ⚠️

**💡 Important Note: Avoid serial I/O operations in loops**

Performing database access, API calls, and other I/O operations in for loops will cause serious performance issues because each operation needs to wait for the previous one to complete, unable to fully utilize async concurrency advantages.

**❌ Wrong Examples: I/O Operations in Loops**

```python
# Wrong: Serial database access in loop
async def get_users_info(user_ids: list[str]):
    results = []
    for user_id in user_ids:
        # Each loop waits for database return, extremely poor performance
        user = await db.users.find_one({"_id": user_id})
        results.append(user)
    return results

# Wrong: Serial API calls in loop
async def fetch_user_profiles(user_ids: list[str]):
    profiles = []
    for user_id in user_ids:
        # Each loop waits for API response, wasting time
        response = await api_client.get(f"/users/{user_id}")
        profiles.append(response.json())
    return profiles

# Wrong: Batch database inserts in loop
async def save_messages(messages: list[dict]):
    for msg in messages:
        # Each message inserted separately, very inefficient
        await db.messages.insert_one(msg)
```

**✅ Correct Examples: Using Concurrent or Batch Operations**

```python
# Correct: Using asyncio.gather for concurrent execution
async def get_users_info(user_ids: list[str]):
    tasks = [db.users.find_one({"_id": uid}) for uid in user_ids]
    results = await asyncio.gather(*tasks)
    return results

# Correct: Using asyncio.gather for concurrent API calls
async def fetch_user_profiles(user_ids: list[str]):
    tasks = [api_client.get(f"/users/{uid}") for uid in user_ids]
    responses = await asyncio.gather(*tasks)
    return [r.json() for r in responses]

# Correct: Using batch insert operation
async def save_messages(messages: list[dict]):
    if messages:
        await db.messages.insert_many(messages)

# Correct: Using database's in query instead of loop queries
async def get_users_info(user_ids: list[str]):
    # Get all data in one query
    cursor = db.users.find({"_id": {"$in": user_ids}})
    results = await cursor.to_list(length=None)
    return results
```

**Performance Comparison**

Assuming 100 users, each database query takes 10ms:
- ❌ Loop serial query: 100 × 10ms = 1000ms (1 second)
- ✅ Concurrent query: ~10ms (almost complete simultaneously)
- ✅ Batch query: ~10ms (single query)

**Exceptional Cases**

In rare cases you may need I/O in loops, but must meet the following conditions:

1. **Subsequent operations depend on previous results**: Must wait for previous operation to complete before proceeding
2. **Rate limiting requirements**: Need to control concurrency to avoid pressure on external services
3. **Already approved by development lead**

```python
# Allowed: Serial operations with dependencies (need comment explaining reason)
async def process_workflow(steps: list[dict]):
    result = None
    for step in steps:
        # Each step depends on previous step's result, cannot be concurrent
        result = await execute_step(step, previous_result=result)
    return result

# Allowed: Using semaphore to control concurrency (need comment explaining reason)
async def fetch_with_rate_limit(urls: list[str]):
    # Limit max 5 concurrent requests to avoid triggering external API rate limit
    semaphore = asyncio.Semaphore(5)
    
    async def fetch_one(url: str):
        async with semaphore:
            return await api_client.get(url)
    
    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks)
```

---

## 🕐 Timezone Awareness Standards

### Core Principles

**💡 Important Note: All time fields must have timezone awareness**

When handling date and time data, must ensure all time fields carry timezone information to avoid data errors and business issues caused by ambiguous timezones.

**⚠️ Prohibit directly using standard methods from `datetime` module**

The project uniformly uses utility functions from `common_utils/datetime_utils.py` to handle time. The following methods are prohibited:
- ❌ `datetime.datetime.now()`
- ❌ `datetime.datetime.utcnow()`
- ❌ `datetime.datetime.today()`

Must use utility functions provided by the project:
- ✅ `get_now_with_timezone()` - Get current time (with timezone)
- ✅ `from_timestamp()` - Convert from timestamp
- ✅ `from_iso_format()` - Convert from ISO format string
- ✅ `to_iso_format()` - Convert to ISO format string
- ✅ `to_timestamp()` / `to_timestamp_ms()` - Convert to timestamp

### Timezone Handling Rules

#### 1. Input Data Timezone Requirements

All time fields entering the system must meet the following requirements:

- **Must carry timezone information**: All datetime type fields must be timezone-aware
- **Default timezone**: If input data doesn't have timezone information, uniformly treat it as **Asia/Shanghai (Shanghai timezone, UTC+8)**
- **Storage format**: When storing in database, recommend uniformly converting to UTC timezone, but must retain timezone information

#### 2. Python Implementation Standards

**✅ Correct Examples: Using Project Utility Functions**

```python
from common_utils.datetime_utils import (
    get_now_with_timezone,
    from_timestamp,
    from_iso_format,
    to_iso_format,
    to_timestamp_ms,
    to_timezone
)

# Method 1: Get current time (automatically with Shanghai timezone)
now = get_now_with_timezone()
# Returns: datetime.datetime(2025, 9, 16, 20, 17, 41, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))

# Method 2: Convert from timestamp (automatically recognize second/millisecond level, automatically add timezone)
dt = from_timestamp(1758025061)
dt_ms = from_timestamp(1758025061000)

# Method 3: Convert from ISO string (automatically handle timezone)
dt = from_iso_format("2025-09-15T13:11:15.588000")  # No timezone, automatically add Shanghai timezone
dt_with_tz = from_iso_format("2025-09-15T13:11:15+08:00")  # Has timezone, keep original timezone then convert

# Method 4: Format to ISO string (automatically include timezone)
iso_str = to_iso_format(now)
# Returns: "2025-09-16T20:20:06.517301+08:00"

# Method 5: Convert to timestamp
ts = to_timestamp_ms(now)
# Returns: 1758025061123
```

**❌ Wrong Examples: Directly using datetime module**

```python
import datetime

# ❌ Wrong: Prohibit using datetime.datetime.now()
naive_dt = datetime.datetime.now()  # Timezone ambiguous, prohibited!

# ❌ Wrong: Prohibit using datetime.datetime.utcnow()
dt = datetime.datetime.utcnow()  # Deprecated in Python 3.12+, prohibited!

# ❌ Wrong: Prohibit using datetime.datetime.today()
dt = datetime.datetime.today()  # Timezone ambiguous, prohibited!

# ❌ Wrong: Manually creating naive datetime
naive_dt = datetime.datetime(2025, 1, 1, 12, 0, 0)  # No timezone information
```

**🔧 How to Fix Existing Code**

```python
# Old code (wrong)
import datetime
now = datetime.datetime.now()

# New code (correct)
from common_utils.datetime_utils import get_now_with_timezone
now = get_now_with_timezone()

# ----------------

# Old code (wrong)
from datetime import datetime
dt = datetime(2025, 1, 1, 12, 0, 0)

# New code (correct)
from common_utils.datetime_utils import from_iso_format
dt = from_iso_format("2025-01-01T12:00:00")  # Automatically add Shanghai timezone

# ----------------

# Old code (wrong)
ts = int(datetime.now().timestamp() * 1000)

# New code (correct)
from common_utils.datetime_utils import get_now_with_timezone, to_timestamp_ms
ts = to_timestamp_ms(get_now_with_timezone())
```

#### 3. Timezone Conversion Examples

```python
from common_utils.datetime_utils import get_now_with_timezone, to_timezone
from zoneinfo import ZoneInfo

# Get Shanghai time
dt_shanghai = get_now_with_timezone()

# Convert to UTC
dt_utc = to_timezone(dt_shanghai, ZoneInfo("UTC"))

# Convert to other timezone
dt_ny = to_timezone(dt_shanghai, ZoneInfo("America/New_York"))
```

#### 4. Database Operation Standards

**MongoDB Example**

```python
from common_utils.datetime_utils import get_now_with_timezone, from_iso_format

# ✅ Correct: Insert time with timezone
data = {
    "created_at": get_now_with_timezone(),
    "updated_at": get_now_with_timezone()
}
await collection.insert_one(data)

# ✅ Correct: Query also use time with timezone
start_time = from_iso_format("2025-01-01T00:00:00")
results = await collection.find({"created_at": {"$gte": start_time}})
```

**PostgreSQL Example**

```python
from common_utils.datetime_utils import get_now_with_timezone

# ✅ Correct: Use timestamptz type
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

# Python query
dt = get_now_with_timezone()
await conn.execute("INSERT INTO events (created_at) VALUES ($1)", dt)
```

#### 5. API Interface Standards

**Receiving External Input**

```python
from common_utils.datetime_utils import from_iso_format
import datetime

def process_datetime_input(dt_str: str) -> datetime.datetime:
    """Process external input time string"""
    try:
        # Use utility function to parse, automatically handle timezone
        # If input has no timezone information, automatically add Shanghai timezone
        dt = from_iso_format(dt_str)
        return dt
    except Exception as e:
        raise ValueError(f"Invalid datetime format: {dt_str}") from e
```

**Returning Output**

```python
from common_utils.datetime_utils import to_iso_format
import datetime

# ✅ Correct: Return ISO 8601 format (including timezone)
def serialize_datetime(dt: datetime.datetime) -> str:
    """Serialize datetime to ISO 8601 format"""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    # Use utility function to format, automatically include timezone information
    return to_iso_format(dt)

# Example output: "2025-01-01T12:00:00+08:00"
```

#### 6. Common Questions and Precautions

**Q: Why choose Shanghai timezone as default timezone?**  
A: The project mainly serves Chinese users, Shanghai timezone (Asia/Shanghai, UTC+8) is the most commonly used timezone.

**Q: Can I use pytz library?**  
A: Python 3.9+ recommends using the standard library's `zoneinfo`, which is the officially recommended timezone handling solution. `pytz` is gradually being phased out.

**Q: Should database store UTC or local timezone?**  
A: Recommend storing UTC timezone, convert to user's required timezone when displaying. This avoids issues like daylight saving time.

**Q: How to handle naive datetime in historical data?**  
A: Need to write data migration script to add Shanghai timezone information to all naive datetimes. Refer to [Data Migration Standards](#data-migration-and-schema-change-process).

### Checklist

During code review, please confirm the following:

- [ ] **Prohibit directly using `datetime.datetime.now()`**, must use `get_now_with_timezone()`
- [ ] **Prohibit directly using `datetime.datetime.utcnow()`** or `datetime.datetime.today()`
- [ ] All time retrieval goes through utility functions in `common_utils/datetime_utils.py`
- [ ] Time parsed from external input uses `from_iso_format()` or `from_timestamp()` for processing
- [ ] Time formatting uses `to_iso_format()` instead of manually calling `.isoformat()`
- [ ] Timestamp conversion uses `to_timestamp_ms()` instead of manual calculation
- [ ] Database schema uses timezone-aware types (e.g. `timestamptz`)
- [ ] Time strings returned by API include timezone information (ISO 8601 format)
- [ ] Test data used in unit tests all have timezone information

---

## 🏛️ Data Access Standards

### Core Principles

**💡 Important Note: All external storage access must go through infra layer repositories**

When handling databases, search engines and other external storage systems, must follow strict layered architecture principles. All data read/write operations must be converged to the `repository` layer in `infra_layer`. Direct calls to external repository capabilities in business layers or other upper layers are prohibited.

**⚠️ Prohibited from directly accessing external storage in the following layers**
- ❌ `biz_layer` (Business layer)
- ❌ `memory_layer` (Memory layer)
- ❌ `agentic_layer` (Agent layer)
- ❌ API interface layer (`api_specs`)
- ❌ Application layer (`app.py`, controllers, etc.)

**✅ Must access through the following methods**
- `infra_layer/adapters/out/persistence/repository/` - Database access
- `infra_layer/adapters/out/search/repository/` - Search engine access

### Why This Standard?

#### 1. Separation of Concerns

Following Hexagonal Architecture and Clean Architecture principles:
- **Business layer**: Focus on business logic, doesn't care where data comes from
- **Infrastructure layer**: Responsible for all interaction details with external systems
- **Isolate changes**: When replacing database or search engine, only need to modify infra layer

#### 2. Testability

```python
# ✅ Benefit: Business layer depends on abstract interface, easy to mock for testing
async def process_user_memory(user_id: str, memory_repo: MemoryRepository):
    """Business logic doesn't depend on concrete implementation"""
    memories = await memory_repo.find_by_user_id(user_id)
    # Business processing...
    
# Can easily replace with mock during testing
mock_repo = MockMemoryRepository()
await process_user_memory("user_1", mock_repo)
```

#### 3. Code Reuse and Consistency

- Avoid repeating same database query logic in multiple places
- Unified handling of exceptions, logging, performance monitoring
- Unified handling of data conversion and validation

#### 4. Centralized Performance Optimization

- Index optimization and query optimization uniformly implemented in repository layer
- Cache strategy uniformly managed
- Batch operation optimization done in one place, benefits entire project

### Correct Architectural Layering

```
┌─────────────────────────────────────────┐
│  API Layer (api_specs, app.py)         │
│  - Receive requests, return responses   │
└─────────────┬───────────────────────────┘
              │ Calls
              ▼
┌─────────────────────────────────────────┐
│  Business Layer (biz_layer)             │
│  - Business logic processing            │
│  - Depends on abstract interfaces (Port)│
└─────────────┬───────────────────────────┘
              │ Dependency Injection
              ▼
┌─────────────────────────────────────────┐
│  Memory Layer (memory_layer)            │
│  - Memory management logic              │
│  - Depends on abstract interfaces (Port)│
└─────────────┬───────────────────────────┘
              │ Dependency Injection
              ▼
┌─────────────────────────────────────────┐
│  Infrastructure Layer (infra_layer)     │
│  - Repository implementations (Adapter) │
│  - Directly operate DB/search engines   │
│  - MongoDB, PostgreSQL, ES, Milvus     │
└─────────────────────────────────────────┘
```

### Implementation Standards

#### ✅ Correct Example: Access through Repository

**Define Repository Interface (Port)**

```python
# core/ports/memory_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional

class MemoryRepository(ABC):
    """Memory repository interface (abstract)"""
    
    @abstractmethod
    async def save(self, memory: Memory) -> str:
        """Save memory"""
        pass
    
    @abstractmethod
    async def find_by_id(self, memory_id: str) -> Optional[Memory]:
        """Find memory by ID"""
        pass
    
    @abstractmethod
    async def find_by_user_id(self, user_id: str, limit: int = 100) -> List[Memory]:
        """Find memory list by user ID"""
        pass
    
    @abstractmethod
    async def search_semantic(self, query: str, user_id: str, top_k: int = 10) -> List[Memory]:
        """Semantic search"""
        pass
```

**Implement Repository (Adapter)**

```python
# infra_layer/adapters/out/persistence/repository/memory_mongo_repository.py
from motor.motor_asyncio import AsyncIOMotorDatabase
from core.ports.memory_repository import MemoryRepository
from core.domain.memory import Memory

class MemoryMongoRepository(MemoryRepository):
    """MongoDB memory repository implementation"""
    
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
        # Call vector search (encapsulated in infra layer)
        # May also call ElasticSearch or Milvus here
        ...
```

**Business Layer Uses Repository**

```python
# biz_layer/services/memory_service.py
from core.ports.memory_repository import MemoryRepository
from core.domain.memory import Memory

class MemoryService:
    """Memory business service"""
    
    def __init__(self, memory_repo: MemoryRepository):
        # ✅ Dependency injection: depend on abstract interface, not concrete implementation
        self._memory_repo = memory_repo
    
    async def create_memory(self, user_id: str, content: str) -> str:
        """Create memory (business logic)"""
        # Business logic: construct domain object
        memory = Memory(user_id=user_id, content=content)
        
        # ✅ Correct: save through repository
        memory_id = await self._memory_repo.save(memory)
        return memory_id
    
    async def get_user_memories(self, user_id: str) -> List[Memory]:
        """Get user memory list"""
        # ✅ Correct: query through repository
        return await self._memory_repo.find_by_user_id(user_id)
    
    async def search_memories(self, user_id: str, query: str) -> List[Memory]:
        """Search memories"""
        # ✅ Correct: semantic search through repository
        return await self._memory_repo.search_semantic(query, user_id)
```

#### ❌ Wrong Example: Business Layer Directly Accesses Database

```python
# ❌ Wrong: Business layer directly uses MongoDB driver
from motor.motor_asyncio import AsyncIOMotorClient

class MemoryService:
    def __init__(self, db_uri: str):
        # ❌ Business layer should not directly connect to database
        self._client = AsyncIOMotorClient(db_uri)
        self._db = self._client["memsys"]
    
    async def create_memory(self, user_id: str, content: str) -> str:
        # ❌ Business layer should not directly operate collection
        result = await self._db.memories.insert_one({
            "user_id": user_id,
            "content": content
        })
        return str(result.inserted_id)
```

```python
# ❌ Wrong: memory_layer directly uses ElasticSearch
from elasticsearch import AsyncElasticsearch

class MemoryRetriever:
    def __init__(self, es_hosts: list):
        # ❌ Should not directly create ES client at this layer
        self._es = AsyncElasticsearch(hosts=es_hosts)
    
    async def search(self, query: str) -> list:
        # ❌ Should not directly call ES API
        result = await self._es.search(index="memories", body={
            "query": {"match": {"content": query}}
        })
        return result["hits"]["hits"]
```

```python
# ❌ Wrong: API layer directly accesses database
from fastapi import APIRouter
from motor.motor_asyncio import AsyncIOMotorClient

router = APIRouter()
db_client = AsyncIOMotorClient("mongodb://localhost")

@router.get("/memories/{user_id}")
async def get_memories(user_id: str):
    # ❌ API layer should not directly query database
    db = db_client["memsys"]
    memories = await db.memories.find({"user_id": user_id}).to_list(100)
    return {"data": memories}
```

### Dependency Injection Configuration

**Use dependency injection container to manage dependencies**

```python
# application_startup.py or bootstrap.py
from dependency_injector import containers, providers
from infra_layer.adapters.out.persistence.repository.memory_mongo_repository import MemoryMongoRepository
from biz_layer.services.memory_service import MemoryService

class Container(containers.DeclarativeContainer):
    """Dependency injection container"""
    
    # Configuration
    config = providers.Configuration()
    
    # Database connection
    mongodb_client = providers.Singleton(
        AsyncIOMotorClient,
        config.mongodb.uri
    )
    
    mongodb_database = providers.Singleton(
        lambda client: client[config.mongodb.database],
        client=mongodb_client
    )
    
    # Repository layer (infrastructure)
    memory_repository = providers.Factory(
        MemoryMongoRepository,
        db=mongodb_database
    )
    
    # Service layer (business logic)
    memory_service = providers.Factory(
        MemoryService,
        memory_repo=memory_repository
    )
```

### Search Engine Access Standards

**ElasticSearch / Milvus also follow Repository pattern**

```python
# infra_layer/adapters/out/search/repository/semantic_memory_es_repository.py
from elasticsearch import AsyncElasticsearch
from typing import List

class SemanticMemoryESRepository:
    """ElasticSearch semantic memory repository"""
    
    def __init__(self, es_client: AsyncElasticsearch, index_name: str):
        self._es = es_client
        self._index = index_name
    
    async def index_memory(self, memory_id: str, content: str, embedding: List[float]):
        """Index memory to ES"""
        await self._es.index(
            index=self._index,
            id=memory_id,
            body={
                "content": content,
                "embedding": embedding
            }
        )
    
    async def search_by_vector(self, query_vector: List[float], top_k: int = 10) -> List[dict]:
        """Vector search"""
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

**Business Layer Calls Search Repository**

```python
# memory_layer/retrievers/semantic_retriever.py
from infra_layer.adapters.out.search.repository.semantic_memory_es_repository import SemanticMemoryESRepository

class SemanticRetriever:
    """Semantic retriever (business logic layer)"""
    
    def __init__(self, search_repo: SemanticMemoryESRepository):
        # ✅ Depend on abstraction, get repository through dependency injection
        self._search_repo = search_repo
    
    async def retrieve_similar_memories(self, query_embedding: List[float], top_k: int = 10):
        """Retrieve similar memories"""
        # ✅ Access search engine through repository
        results = await self._search_repo.search_by_vector(query_embedding, top_k)
        # Business logic: filtering, sorting, formatting, etc.
        return self._process_results(results)
```

### Multiple Data Source Scenarios

**Repository can encapsulate access to multiple data sources**

```python
# infra_layer/adapters/out/persistence/repository/memory_hybrid_repository.py
class MemoryHybridRepository(MemoryRepository):
    """Hybrid memory repository: MongoDB + ElasticSearch"""
    
    def __init__(
        self,
        mongo_repo: MemoryMongoRepository,
        es_repo: SemanticMemoryESRepository
    ):
        self._mongo = mongo_repo
        self._es = es_repo
    
    async def save(self, memory: Memory) -> str:
        """Save to MongoDB and ES"""
        # Save to MongoDB
        memory_id = await self._mongo.save(memory)
        
        # Sync to ElasticSearch (async task or immediate sync)
        await self._es.index_memory(
            memory_id=memory_id,
            content=memory.content,
            embedding=memory.embedding
        )
        
        return memory_id
    
    async def search_semantic(self, query: str, user_id: str, top_k: int = 10) -> List[Memory]:
        """Semantic search: ES query + MongoDB supplement details"""
        # 1. ES search to get relevant IDs
        es_results = await self._es.search_by_text(query, top_k)
        memory_ids = [hit["_id"] for hit in es_results]
        
        # 2. MongoDB batch query for complete data
        memories = await self._mongo.find_by_ids(memory_ids)
        return memories
```

### Checklist

When writing or reviewing code, please confirm the following:

- [ ] **Are database operations in infra_layer/repository?**
- [ ] **Are search engine operations in infra_layer/repository?**
- [ ] **Does business layer depend on abstract interfaces (Port) rather than concrete implementations?**
- [ ] **Is dependency injection used to pass repositories?**
- [ ] **Avoid directly creating database connections in business/API/application layers?**
- [ ] **Avoid directly using MongoDB/PostgreSQL/ES/Milvus clients in business layer?**
- [ ] **Are newly added Repositories registered in dependency injection container?**
- [ ] **Do Repository methods have clear business semantics (rather than exposing underlying implementation details)?**

### Common Questions

**Q: Why can't I directly use MongoDB driver in business layer?**  
A: It violates architectural layering principles, causing business logic to be coupled with infrastructure, making it difficult to test and replace data sources.

**Q: Do simple queries also need to go through Repository?**  
A: Yes. Even simple queries should be encapsulated in Repository. This allows:
   - Unified management of all data access
   - Only need to modify one place for subsequent optimization
   - Keep code style consistent

**Q: Should Repository methods return dict or domain objects?**  
A: Recommend returning domain objects (like `Memory`, `User`), so business layer doesn't need to care about underlying data format.

**Q: How to handle complex join queries?**  
A: Encapsulate complex query logic in Repository layer, provide semantic methods externally. For example:
```python
async def find_memories_with_user_info(self, user_id: str) -> List[MemoryWithUser]:
    # Handle join or multiple queries inside Repository
    ...
```

**Q: Can I call other Repositories within a Repository?**  
A: Yes, but be careful:
   - Avoid circular dependencies
   - Complex cross-data-source logic should be coordinated in business layer
   - Repository responsibilities should be single

### Related Documentation

- [Hexagonal Architecture](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software))
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Dependency Injection Pattern](https://python-dependency-injector.ets-labs.org/)

---

## 📥 Import Standards

### PYTHONPATH Management

**💡 Important Note: PYTHONPATH needs unified management**

The project uniformly manages `PYTHONPATH` and module import paths. Changes involving path configuration should be communicated with development lead before unified configuration.

#### Why Need Unified Management?

- Chaotic import paths may cause modules not found or import errors
- Inconsistent paths in different environments (development/test/production) may cause deployment issues
- Inconsistent IDE configuration may affect team collaboration
- Mixing relative imports and absolute imports increases code maintenance difficulty

#### Management Scope

Import paths for the following directories in the project should be kept consistent:

- `src/`: Main business code
- `tests/`: Test code
- `unit_test/`: Unit tests
- `evaluation/`: Evaluation scripts
- Other directories that need to be imported (such as `demo/` etc.)

#### Recommended Practices

1. **Unified Project Root Directory**
   - Project root directory is `/Users/admin/memsys` (or corresponding path in deployment environment)
   - src directory is added to PYTHONPATH, import starts directly from module name

2. **Import Standard Examples**

```python
# ✅ Recommended: Absolute import (src already in PYTHONPATH)
from core.memory.manager import MemoryManager
from infra_layer.adapters.out.db import MongoDBAdapter
from tests.fixtures.mock_data import get_mock_user

# ✅ Recommended: Import in test files
from unit_test.email_data_constructor import construct_email

# ❌ Not recommended: Relative import across levels
from ...core.memory.manager import MemoryManager

# ❌ Not recommended: Including src prefix (src already in PYTHONPATH, no prefix needed)
from src.core.memory.manager import MemoryManager

# ❌ Not recommended: Using sys.path.append to temporarily modify path
import sys
sys.path.append("../src")  # May cause environment inconsistency
```

3. **Path Configuration Change Process**

If you need to:
- Add new importable directory
- Modify existing directory import method
- Adjust PYTHONPATH configuration

**Recommended process**:

1. **Discuss with Development Lead**: Explain reason and impact scope of change
2. **Unified Configuration**: Update the following files
   - `src/bootstrap.py`: Startup entry
   - `auto.sh`: Automation scripts
   - `.vscode/settings.json` or `.idea` configuration
   - Environment variable settings in deployment scripts
3. **Document Update**: Update this document and related development documents
4. **Team Notification**: Notify all developers to sync configuration

4. **IDE Configuration (Recommended Unified)**

Recommend marking project root directory as Sources Root in IDE:

- **PyCharm**: Right-click project root directory → Mark Directory as → Sources Root
- **VSCode**: Configure in `.vscode/settings.json`:
  ```json
  {
    "python.analysis.extraPaths": [
      "${workspaceFolder}"
    ]
  }
  ```

### Prefer Absolute Imports

**💡 Important Note: Recommend using absolute imports, avoid relative imports**

#### Why Recommend Absolute Imports?

Relative imports, although more concise in some scenarios, have the following issues:

- **Poor readability**: `from ...core.memory import Manager` is not as intuitive as `from core.memory import Manager`
- **Difficult refactoring**: Moving files requires modifying all relative import levels
- **Complex debugging**: Relative import paths in stack traces are not clear enough
- **Tool support**: IDE and static analysis tools have better support for absolute imports
- **Test convenience**: Test files using absolute imports are easier to understand dependency relationships

#### Import Method Comparison

```python
# ✅ Recommended: Absolute import (src already in PYTHONPATH)
from core.memory.manager import MemoryManager
from core.memory.types import MemoryType, MemoryStatus
from infra_layer.adapters.out.db.mongodb import MongoDBAdapter
from common_utils.logger import get_logger

# ✅ Acceptable: Relative import within same package (single level)
# File: src/core/memory/manager.py
from .types import MemoryType  # Same directory
from .extractors.base import BaseExtractor  # Subdirectory

# ❌ Not recommended: Relative import across levels
from ...infra_layer.adapters import MongoDBAdapter
from ....common_utils.logger import get_logger

# ❌ Not recommended: Relative import up multiple levels (hard to maintain)
from ......some_module import something
```

#### Usage Rules

**Recommended practices**:

1. **Cross-module imports must use absolute imports**
   ```python
   # From src/core/memory/manager.py import to src/biz_layer/service.py
   from core.memory.manager import MemoryManager  # ✅
   ```

2. **Within same package can use single-level relative import**
   ```python
   # In src/core/memory/manager.py
   from .types import MemoryType  # ✅ Same directory
   from .extractors.base import BaseExtractor  # ✅ Subdirectory
   ```

3. **Avoid relative imports up multiple levels**
   ```python
   from ...infra_layer import something  # ❌ Should change to absolute import
   from infra_layer import something  # ✅
   ```

#### Special Scenario Descriptions

**Scenario 1: Module imports within package**

For modules within a package that need to import each other:

```python
# Package structure:
# src/core/memory/
#   ├── __init__.py
#   ├── manager.py
#   ├── types.py
#   └── extractors/
#       ├── __init__.py
#       └── base.py

# In manager.py:
from .types import MemoryType  # ✅ Acceptable single-level relative import
from core.memory.types import MemoryType  # ✅ Can also use absolute import

# In extractors/base.py:
from ..types import MemoryType  # 🤔 OK, but absolute import is better
from core.memory.types import MemoryType  # ✅ Recommended
```

**Scenario 2: Test file imports**

Test files recommend using absolute imports entirely:

```python
# tests/test_memory_manager.py
from core.memory.manager import MemoryManager  # ✅
from core.memory.types import MemoryType  # ✅
from tests.fixtures.mock_data import get_mock_data  # ✅
```

### __init__.py Usage Standards

**💡 Important Note: Not recommended to write any code in `__init__.py`**

#### Why Keep `__init__.py` Empty?

- **Import side effects**: `__init__.py` executes when package is imported, any code may produce unexpected side effects
- **Circular dependencies**: Even simple module exports can easily lead to circular import issues
- **Performance impact**: Executing code when importing package affects startup performance and module loading speed
- **Maintainability**: Code scattered in `__init__.py` is hard to locate and maintain
- **Testing difficulty**: Mock and unit testing become complex
- **Implicit behavior**: Implicit execution during import increases code understanding difficulty

#### Recommended Usage

**✅ Recommended: Keep empty file**

```python
# src/core/memory/__init__.py

# Empty file, only serves as Python package identifier
# Don't write any code here
```

**How to import modules?**

Import directly from specific module files, don't depend on re-export from `__init__.py`:

```python
# ✅ Recommended: Import directly from module file
from core.memory.manager import MemoryManager
from core.memory.types import MemoryType, MemoryStatus
from core.memory.extractors.base import BaseExtractor

# ❌ Not recommended: Depend on re-export from __init__.py
from core.memory import MemoryManager  # Requires export code in __init__.py
```

**❌ Not Recommended Practices**

```python
# ❌ Don't re-export modules in __init__.py
# src/core/memory/__init__.py
from .manager import MemoryManager
from .types import MemoryType, MemoryStatus
from .extractors import BaseExtractor

__all__ = ["MemoryManager", "MemoryType", "MemoryStatus", "BaseExtractor"]
# Although looks harmless, it increases circular dependency risk and maintenance cost
```

```python
# ❌ Don't initialize global objects in __init__.py
# src/core/memory/__init__.py
from .manager import MemoryManager

# Don't do this!
global_memory_manager = MemoryManager()  # ❌
config = load_config()  # ❌
db_connection = connect_to_db()  # ❌
```

```python
# ❌ Don't write business functions or classes in __init__.py
# src/core/memory/__init__.py

def process_memory(data):  # ❌ Should be in separate module file
    # Business logic...
    pass

class MemoryProcessor:  # ❌ Should be in separate module file
    pass
```

```python
# ❌ Don't execute any logic in __init__.py
# src/core/__init__.py

# Don't do this!
import logging
logging.basicConfig(...)  # ❌ Side effects

if some_condition:  # ❌ Conditional execution
    do_something()

for item in items:  # ❌ Loop logic
    process(item)

__version__ = "1.0.0"  # ❌ Even version information is not recommended
```

#### Correct Code Organization

Keep `__init__.py` empty, put all business code in separate module files:

```python
# Package structure:
# src/core/memory/
#   ├── __init__.py          # Empty file
#   ├── manager.py           # MemoryManager class
#   ├── types.py             # Type definitions
#   ├── processor.py         # Business processing functions
#   └── config.py            # Configuration management

# src/core/memory/__init__.py - Keep empty
# (Empty file, don't write any code)

# src/core/memory/manager.py - Actual business logic
class MemoryManager:
    def __init__(self):
        # Implementation details...
        pass

# src/core/memory/processor.py - Business functions
def process_memory(data):
    # Actual business logic...
    pass

# When using, import directly from specific module
from core.memory.manager import MemoryManager
from core.memory.types import MemoryType, MemoryStatus
from core.memory.processor import process_memory
```

#### Checklist

When writing or reviewing `__init__.py`, please confirm:

- [ ] Is the file empty (or only contains comments)?
- [ ] Are there no import statements?
- [ ] Are there no defined variables or constants?
- [ ] Are there no created global object instances?
- [ ] Are there no defined classes or functions?
- [ ] Is there no executed logic?

**If any of the above answers "no", please move the code to a separate module file.**

---

## 🌿 Branch Management Standards

### Branch Type Descriptions

| Branch | Description | Notes |
|------|------|------|
| `master` | Stable version; only for bug fix checkout, `release/xxx` and `hotfix/xxx` merge here | Production environment deployment branch |
| `dev` | Daily development version; continuous code commits | If versioning has started & commit is for current version, commit to `release`; non-urgent small bugs & features merge to `dev`, catch next release |
| `release/YYMMDD` | Version branch; deploy to test first, then production; first `dev` merges `master`, then cut from `dev`; after actual release, merge back to `master`, `dev` | Currently irregular (notified in group); only bug commits for current release |
| `feature/xxxx` | Single cycle, small feature; merge to `dev` or a `release` | Can directly merge to `dev`; recommend MR for merge to `release` |
| `bugfix/xxxx` | Single cycle, small bug; merge to `dev` or a `release` | Can directly merge to `dev`; recommend MR for merge to `release` |
| `long/xxx` | Cross-cycle, large feature; cut from `dev`, merge to `dev` or a `release` | Test separately in new test environment (distinguished by port/address); regularly merge `dev` to avoid too many conflicts at end; recommend MR |
| `hotfix/xxxx` | Bug fix; cut from `master`, MR to `master` branch (also to `dev` if needed) | Only exists after release; bugs during normal development directly merge to `dev`, during versioning but not released yet merge to `release`, only use this for urgent bugs when not versioning; recommend MR |

### Environment and Branch Correspondence

| Environment | Possible Branches | Notes |
|------|----------|------|
| Production | `master` branch | Stable version |
|      | `release/xxx` branch | After versioned release and before bug fix |
| Test | `dev` branch | Daily development stage |
|      | `release/xxx` branch | Versioned testing stage |
|      | `hotfix/xxxx` | Emergency bug fix |

### Version Tag Standards

| Tag | Description | Notes |
|-----|------|------|
| `X.Y.Z` | Version number: Major.Iteration.BugFix | Not necessarily synced with iterations, add when needed |

- **X (Major version)**: Major architecture changes or incompatible updates
- **Y (Iteration version)**: Feature iterations, new feature additions
- **Z (Fix version)**: Bug fixes, minor optimizations

### Branch Operation Process

#### 1. Daily Development (feature/bugfix)

```bash
# Create feature branch from dev
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name

# After development complete
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name

# Merge to dev (small features can directly merge)
git checkout dev
git merge feature/your-feature-name
git push origin dev

# Delete feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

#### 2. Release Process (release)

```bash
# 1. Let dev merge master first (ensure including latest hotfixes)
git checkout dev
git pull origin dev
git merge master
git push origin dev

# 2. Create release branch from dev
git checkout -b release/$(date +%y%m%d)
git push origin release/$(date +%y%m%d)

# 3. Bug fixes during testing phase
git checkout release/$(date +%y%m%d)
# ... fix bugs ...
git commit -m "fix: bug description"
git push origin release/$(date +%y%m%d)

# 4. Merge back to master and dev after release
git checkout master
git merge release/$(date +%y%m%d)
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin master --tags

git checkout dev
git merge release/$(date +%y%m%d)
git push origin dev
```

#### 3. Emergency Fix (hotfix)

```bash
# Create hotfix branch from master
git checkout master
git pull origin master
git checkout -b hotfix/critical-bug-fix

# After fix complete, recommend MR process
git add .
git commit -m "hotfix: critical bug description"
git push origin hotfix/critical-bug-fix

# Create Merge Request to master
# After merge, remember to sync to dev
git checkout dev
git merge master
git push origin dev
```

#### 4. Long-term Feature Development (long)

```bash
# Create long-term branch from dev
git checkout dev
git pull origin dev
git checkout -b long/big-feature

# Regularly merge dev to avoid conflict accumulation
git checkout long/big-feature
git merge dev

# After feature complete, recommend MR process to merge to dev or release
```

### Unified Branch Merge Handling Standards

**⚠️ Important Note: The following branch merge operations need to be uniformly handled by development or operations lead**

To ensure code quality and standard release process, the following types of branch merge operations need to be uniformly managed and executed by development lead or operations lead:

#### Merge Operations Requiring Unified Handling

1. **Long-term feature branch merge to dev**
   - `long/xxx` → `dev`
   - Reason: Long-term feature branches usually involve large code changes, need to assess impact scope and potential conflicts

2. **Dev cut out release branch**
   - `dev` → `release/YYMMDD`
   - Reason: Release nodes need unified coordination to ensure version content is complete and meets release requirements

3. **Release merge back to dev**
   - `release/YYMMDD` → `dev`
   - Reason: Ensure bug fixes from release branch can be correctly synced back to main development branch

#### Operation Process

**Developer Operations**:
```bash
# 1. Ensure branch code is pushed to remote
git push origin your-branch

# 2. Contact development lead or operations lead
# 3. Explain merge requirements:
#    - Source branch name
#    - Target branch name
#    - Merge reason and impact scope
#    - Whether testing is complete
```

**Lead Operations**:
```bash
# 1. Check code quality and testing situation
# 2. Assess conflicts and impact scope
# 3. Choose appropriate time window to execute merge
# 4. Notify relevant personnel after merge complete
```

#### Why Need Unified Handling?

- **Code quality control**: Ensure merged code has gone through sufficient testing and review
- **Version management standards**: Avoid version confusion and non-standard release process
- **Professional conflict handling**: Complex conflicts need experienced personnel to handle
- **Unified team coordination**: Avoid chaos caused by multiple people operating simultaneously
- **Risk control**: Important branch merges need rollback plans

#### Precautions

- Small feature branches (`feature/xxx`, `bugfix/xxx`) merging to `dev` can be done by developers themselves
- Emergency `hotfix` merging to `master` recommend MR process and review by lead
- All merges involving `release` and `master` recommend confirmation by lead

---

## 🔍 Code Review Process

### Code Review Recommended Scenarios

The following situations recommend Code Review (creating Merge Request/Pull Request):

#### 1. Data-related Changes

- 💾 **Database Migration Scripts**
  - New additions or modifications in `migrations/` directory
  - Data structure changes (field additions/deletions/modifications, index changes)
  
- 💾 **Data Fix Scripts**
  - Scripts in `devops_scripts/data_fix/` directory
  - Scripts involving batch data modifications
  - ElasticSearch data sync scripts

#### 2. Dependency Package Changes

- 📦 **Dependency Package Additions**
  - Adding production or development dependencies
  - Explain reason and purpose for addition
  
- 📦 **Dependency Package Deletions**
  - Confirm no other modules depend on it
  - Explain reason for deletion
  
- 📦 **Dependency Package Version Upgrades**
  - Major version upgrades (e.g. 1.x → 2.x)
  - Upgrades involving API changes
  - Provide upgrade impact assessment

Related changes affect:
- `pyproject.toml`
- `uv.lock`

#### 3. Infrastructure Changes

- 🏗️ **Infrastructure Additions or Changes**
  - Adding middleware, database connection pools and other basic components
  - Modifying infrastructure configuration (database, Redis, Kafka, etc.)
  - Adding or modifying adapters under `infra_layer/`
  
- 🏗️ **Application Startup Loading Process**
  - Modifying `bootstrap.py`
  - Modifying `application_startup.py`
  - Modifying `base_app.py`
  - Modifying dependency injection container configuration

#### 4. Important Branch Merges

- 🔀 Merge to `release/xxx` branch
- 🔀 Merge to `master` branch (hotfix)
- 🔀 `long/xxx` long-term branch merge to `dev`

### Data Migration and Schema Change Process

**⚠️ Important Principle: Plan ahead, communicate fully**

When launching new features involving data fixes or Schema migration, should discuss with development lead and operations lead as early as possible to ensure feasibility of data migration solution and timing arrangement for subsequent implementation.

#### Why Need Early Communication?

Data migration and Schema changes are high-risk operations that may affect:

- **Data integrity**: Data structure changes may cause data loss or corruption
- **Service availability**: Large-scale data migration may affect service performance
- **Rollback complexity**: Rollback after Schema changes is often more complex than code rollback
- **Time window**: Need to reserve enough time for data migration and verification
- **Multi-team collaboration**: Involves cooperation of multiple teams including development, testing, and operations

#### Data Migration Process Recommendations

##### 1. Solution Design Phase

**Timing**: Before feature development or early in development

**Operation Steps**:
- Discuss with development lead:
  - Necessity of data structure changes
  - Technical feasibility of migration solution
  - Whether need to be compatible with old data format
  - Estimate data volume and migration duration
  
- Discuss with operations lead:
  - Impact of migration on service performance
  - Whether need downtime maintenance
  - Rollback solution and contingency plan
  - Data backup strategy

**Deliverables**:
- Data migration solution document
- Risk assessment report
- Time schedule plan

##### 2. Script Development Phase

**Timing**: Mid-feature development

**Operation Steps**:
- Write Migration script or data fix script
- Test script in development environment
- Assess script performance (processing speed, resource usage)
- Prepare data verification script
- Write rollback script

**Precautions**:
- Script must go through Code Review (see "Data-related Changes" above)
- Script should support incremental execution and checkpoint resume
- Add detailed logging
- Consider batch processing, avoid loading large amounts of data at once

##### 3. Test Verification Phase

**Timing**: Late feature development

**Operation Steps**:
- Execute complete migration process in test environment
- Verify data integrity and correctness
- Test compatibility of new features with migrated data
- Test effectiveness of rollback process
- Record migration duration and resource usage

**Verification Checklist**:
- [ ] Data volume statistics correct (before and after migration comparison)
- [ ] Data format meets expectations
- [ ] Indexes correctly established
- [ ] New features run normally
- [ ] Old data compatibility verification passed
- [ ] Rollback script test successful

##### 4. Production Implementation Phase

**Timing**: Confirm with operations before release day

**Operation Steps**:

1. **Confirm implementation time with operations**
   - Determine migration window (avoid business peak hours)
   - Confirm if downtime maintenance needed
   - Notify relevant parties (product, customer service, etc.)

2. **Preparation**
   - Backup production data
   - Prepare monitoring scripts
   - Prepare rollback scripts
   - Prepare contingency plan

3. **Execute Migration**
   - Execute according to established plan
   - Real-time monitor migration progress
   - Record detailed logs

4. **Verification and Monitoring**
   - Verify data migration results
   - Monitor service running status
   - Observe performance metrics
   - Collect user feedback

##### 5. Follow-up Tracking

**Timing**: 1-3 days after launch

**Operation Steps**:
- Continuously monitor data and service status
- Handle remaining issues
- Summarize migration experience
- Update documentation

#### Common Scenario Examples

| Scenario | Early Communication Time | Key Discussion Points |
|------|--------------|------------|
| **Add Field** | Early development (1-2 weeks before) | Default value strategy, index establishment, whether need to backfill historical data |
| **Field Type Change** | Solution design stage (2-3 weeks before) | Data conversion rules, incompatible data handling, rollback solution |
| **Large-scale Data Fix** | Solution design stage (2-4 weeks before) | Data volume assessment, migration duration, batch strategy, downtime plan |
| **Index Rebuild** | Solution design stage (1-2 weeks before) | Performance impact, execution time window, online/offline method |
| **Data Archive/Cleanup** | Solution design stage (2-3 weeks before) | Archive strategy, data backup, recovery mechanism |

#### Related Documentation

- [MongoDB Migration Guide](./mongodb_migration_guide.md)
- Data fix script directory: `src/devops_scripts/data_fix/`
- Migration script directory: `src/migrations/`

### Code Review Process

#### Submitter Recommendations

1. **Create Merge Request**
   - Fill in clear title and description
   - Explain reason for change and impact scope
   - Link related Issues or requirements

2. **Self-check Checklist**
   - [ ] Code passes pre-commit checks
   - [ ] Related unit tests added/updated
   - [ ] Documentation updated (if necessary)
   - [ ] No obvious performance issues
   - [ ] No security risks

3. **Assign Reviewers**
   - Assign reviewers in Merge Request (see contacts at end of document)
   - Clearly mark change type:
     - 📊 Data change (Migration/data fix script)
     - 📦 Dependency change (pyproject.toml/uv.lock)
     - 🏗️ Infrastructure change (infra_layer/startup process)
     - 🔀 Important branch merge (release/master/long)

#### Reviewer Work

1. **Code Quality Review**
   - Code logic correctness
   - Code readability and maintainability
   - Whether conforms to project standards

2. **Risk Assessment**
   - Data security risk (especially focus on data scripts)
   - Performance impact (async code, database queries)
   - Compatibility issues (dependency upgrades, API changes)

3. **Review Feedback**
   - Provide clear modification suggestions
   - Mark severity (Must Fix / Should Fix / Nice to Have)
   - Timely response (within 24 hours if possible)

### MR Description Template

```markdown
## Change Type
- [ ] Feature (new feature)
- [ ] Bugfix (bug fix)
- [ ] Hotfix (emergency fix)
- [ ] Refactor (refactoring)
- [ ] 📊 Migration (data migration) - recommend Code Review
- [ ] 📦 Dependency (dependency change) - recommend Code Review
- [ ] 🏗️ Infrastructure (infrastructure change) - recommend Code Review

## Reviewers
<!-- Assign reviewers, see contact information at end of document -->

## Change Description
<!-- Briefly describe content and reason for this change -->

## Impact Scope
<!-- Explain affected modules, services or features -->

## Testing Status
- [ ] Code passes pre-commit checks
- [ ] Unit tests passed
- [ ] Integration tests passed
- [ ] Manual testing completed

## Risk Assessment
<!-- If involving data/dependency/infrastructure changes, explain risks and rollback plan -->

## Related Documentation
<!-- Link to requirement documents, design documents or Issues -->

## Screenshots/Logs
<!-- If necessary, provide screenshots or key logs -->
```

### Which Changes Recommend Code Review?

The following situations recommend creating MR and assigning reviewers:

- 💾 **Data-related changes** (Migration, data fix scripts)
- 📦 **Dependency package additions/deletions/modifications** (pyproject.toml, uv.lock)
- 🏗️ **Infrastructure changes** (infra_layer, startup process)
- 🔀 **Merge to release/xxx or master branch**
- 🔀 **long/xxx long-term branch merge to dev**

**Small features/small bugs merging to dev** can decide whether MR is needed on a case-by-case basis.

### How to Submit Merge Request?

```bash
# 1. Push your branch to remote
git push origin your-branch-name

# 2. Create MR/PR on Git platform (GitLab/GitHub/Gitee)
#    - Source branch: your-branch-name
#    - Target branch: dev/release/master
#    - Reviewers: see contacts at end of document
#    - Use MR description template above to fill in detailed information

# 3. Wait for review, modify code based on feedback
git add .
git commit -m "fix: modify xxx according to code review feedback"
git push origin your-branch-name

# 4. Merge after review approval
#    - Merge after reviewer confirms
#    - Or merge yourself and notify relevant personnel
```

---

## 📚 Related Documentation

- [Getting Started Guide](./getting_started.md)
- [Development Guide](./development_guide.md)
- [Dependency Management Guide](./project_deps_manage.md)
- [Bootstrap Usage Instructions](./bootstrap_usage.md)
- [MongoDB Migration Guide](./mongodb_migration_guide.md)

---

## ❓ FAQ

### Q1: What if I forgot to install pre-commit hook?

```bash
pre-commit install
pre-commit run --all-files  # Run checks on existing code
```

### Q2: What if I accidentally installed a package with pip?

```bash
# 1. Uninstall package installed with pip
pip uninstall <package-name>

# 2. Reinstall with uv
uv add <package-name>

# 3. Re-sync environment
uv sync
```

### Q3: What to do with branch merge conflicts?

```bash
# 1. Ensure local branch is up to date
git checkout your-branch
git pull origin your-branch

# 2. Merge target branch
git merge target-branch

# 3. Commit after resolving conflicts
git add .
git commit -m "merge: resolve conflicts with target-branch"
```

### Q4: How to call sync library in async code?

```python
import asyncio

async def use_sync_library():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Use default thread pool
        sync_function,
        arg1,
        arg2
    )
    return result
```

---

## 👤 Contacts

### Development Lead

For the following matters, recommend communicating with development lead:

- Thread/process usage solution discussion
- PYTHONPATH path configuration changes
- Code Review requests
- Technical solutions for data scripts, dependency changes, infrastructure modifications

**Current Lead**: zhanghui

### Operations Lead

For the following matters, please contact operations lead:

- Obtain development environment configuration (database, middleware connection information)
- Service access permission application
- Environment configuration problem troubleshooting
- New configuration items or environment requirements
- Network connection, VPN and other infrastructure issues

**Current Lead**: jianhua

---

**Last Updated**: 2025-10-31
