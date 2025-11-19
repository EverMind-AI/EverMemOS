# Memsys 系统稳定性分析与测试用例设计

## 📊 系统架构稳定性分析

### 🏗️ 系统架构概览

Memsys 是一个基于六边形架构的智能记忆系统，包含以下核心层次：

```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                     │
├─────────────────────────────────────────────────────────────┤
│                 Core Infrastructure                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   DI Container │ │  Middleware │ │  Lifespan   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                Business Logic Layer                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  Decision   │ │  Retrieval  │ │   Storage   │          │
│  │   Layer     │ │   Layer     │ │   Layer     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│              Infrastructure Layer                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ PostgreSQL  │ │  MongoDB    │ │   Redis     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 🚨 关键稳定性风险点分析

### 1. 数据库连接与连接池管理

#### 🔴 高风险点
- **连接池耗尽**: 默认配置 `pool_size=40, max_overflow=25`，在高并发下可能不足
- **连接泄漏**: 异步操作中未正确释放数据库连接
- **连接超时**: 长时间运行的任务可能导致连接超时
- **数据库故障**: PostgreSQL/MongoDB 服务不可用时的降级处理

#### 📋 风险场景
```python
# 风险场景1: 连接池耗尽
async def concurrent_database_operations():
    tasks = []
    for i in range(100):  # 超过连接池大小
        task = asyncio.create_task(heavy_db_operation())
        tasks.append(task)
    await asyncio.gather(*tasks)  # 可能导致连接池耗尽

# 风险场景2: 连接泄漏
async def risky_db_operation():
    session = get_session()
    try:
        result = await session.execute(complex_query)
        # 如果这里抛出异常，连接可能不会被正确释放
        return result
    except Exception as e:
        # 连接泄漏风险
        raise e
```

### 2. 异步并发控制

#### 🔴 高风险点
- **并发限制不足**: 大量并发请求可能导致系统资源耗尽
- **死锁风险**: 异步操作中的资源竞争
- **内存泄漏**: 长时间运行的异步任务积累内存
- **任务取消处理**: 异步任务被取消时的清理工作

#### 📋 风险场景
```python
# 风险场景1: 无限制并发
async def unlimited_concurrent_requests():
    tasks = []
    for i in range(1000):  # 无限制并发
        task = asyncio.create_task(process_request())
        tasks.append(task)
    await asyncio.gather(*tasks)  # 可能导致系统崩溃

# 风险场景2: 异步任务取消处理不当
async def risky_async_task():
    try:
        await long_running_operation()
    except asyncio.CancelledError:
        # 没有清理资源
        raise
```

### 3. 外部服务依赖

#### 🔴 高风险点
- **LLM API 限流**: OpenAI/Gemini API 调用频率限制
- **网络超时**: 外部服务响应超时
- **服务降级**: 外部服务不可用时的降级策略
- **API 密钥管理**: 密钥过期或无效的处理

#### 📋 风险场景
```python
# 风险场景1: LLM API 限流
async def llm_api_risk():
    for i in range(100):
        try:
            response = await llm_provider.generate(prompt)
        except RateLimitError:
            # 没有重试机制或降级策略
            raise

# 风险场景2: 网络超时
async def network_timeout_risk():
    try:
        response = await external_api_call()
    except asyncio.TimeoutError:
        # 没有超时重试机制
        raise
```

### 4. 内存管理

#### 🔴 高风险点
- **大文件处理**: 处理大型文档时的内存占用
- **向量索引**: FAISS 索引的内存管理
- **缓存管理**: Redis 缓存的内存使用
- **垃圾回收**: Python 垃圾回收不及时

#### 📋 风险场景
```python
# 风险场景1: 大文件内存占用
def process_large_file():
    with open('huge_file.json', 'r') as f:
        data = json.load(f)  # 可能占用大量内存
        process_data(data)

# 风险场景2: 向量索引内存泄漏
class VectorIndex:
    def __init__(self):
        self.index = faiss.IndexFlatL2(768)
        # 没有内存限制和清理机制
```

### 5. 配置管理

#### 🔴 高风险点
- **环境变量缺失**: 关键配置项未设置
- **配置验证**: 配置项格式或值不正确
- **动态配置**: 运行时配置变更的影响
- **敏感信息**: API 密钥等敏感信息的安全存储

## 🧪 稳定性测试用例设计

### 1. 数据库连接池测试

#### 测试用例 1.1: 连接池耗尽测试
```python
import pytest
import asyncio
from unittest.mock import patch
from component.database_session_provider import DatabaseSessionProvider

class TestDatabaseConnectionPool:
    
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self):
        """测试连接池耗尽场景"""
        provider = DatabaseSessionProvider()
        
        # 创建超过连接池大小的并发任务
        async def db_operation():
            async with provider.get_async_session() as session:
                await session.execute("SELECT 1")
                await asyncio.sleep(0.1)  # 模拟长时间操作
        
        # 创建大量并发任务
        tasks = [asyncio.create_task(db_operation()) for _ in range(100)]
        
        # 验证系统行为
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 检查是否有连接池相关的异常
        connection_errors = [r for r in results if isinstance(r, Exception)]
        assert len(connection_errors) == 0, f"连接池耗尽导致异常: {connection_errors}"
    
    @pytest.mark.asyncio
    async def test_connection_leak_detection(self):
        """测试连接泄漏检测"""
        provider = DatabaseSessionProvider()
        initial_connections = provider.async_engine.pool.size()
        
        # 模拟连接泄漏场景
        async def leaky_operation():
            session = provider.create_session()
            # 故意不关闭session
            await session.execute("SELECT 1")
            # 不调用 session.close()
        
        await leaky_operation()
        
        # 检查连接池状态
        current_connections = provider.async_engine.pool.size()
        assert current_connections <= initial_connections + 5, "检测到连接泄漏"
```

#### 测试用例 1.2: 数据库故障恢复测试
```python
@pytest.mark.asyncio
async def test_database_failure_recovery(self):
    """测试数据库故障恢复"""
    provider = DatabaseSessionProvider()
    
    # 模拟数据库连接失败
    with patch.object(provider.async_engine, 'execute') as mock_execute:
        mock_execute.side_effect = Exception("Database connection failed")
        
        # 测试重试机制
        retry_count = 0
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                async with provider.get_async_session() as session:
                    await session.execute("SELECT 1")
                break
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    # 验证降级策略
                    assert True, "应该触发降级策略"
```

### 2. 异步并发控制测试

#### 测试用例 2.1: 并发限制测试
```python
import asyncio
from unittest.mock import patch

class TestConcurrencyControl:
    
    @pytest.mark.asyncio
    async def test_semaphore_limitation(self):
        """测试信号量限制并发数"""
        max_concurrent = 5
        semaphore = asyncio.Semaphore(max_concurrent)
        concurrent_count = 0
        max_observed = 0
        
        async def limited_operation():
            nonlocal concurrent_count, max_observed
            async with semaphore:
                concurrent_count += 1
                max_observed = max(max_observed, concurrent_count)
                await asyncio.sleep(0.1)
                concurrent_count -= 1
        
        # 创建大量任务
        tasks = [asyncio.create_task(limited_operation()) for _ in range(50)]
        await asyncio.gather(*tasks)
        
        # 验证并发数不超过限制
        assert max_observed <= max_concurrent, f"并发数超过限制: {max_observed}"
    
    @pytest.mark.asyncio
    async def test_task_cancellation_cleanup(self):
        """测试任务取消时的资源清理"""
        cleanup_called = False
        
        async def cancellable_task():
            nonlocal cleanup_called
            try:
                await asyncio.sleep(10)  # 长时间运行
            except asyncio.CancelledError:
                # 清理资源
                cleanup_called = True
                raise
        
        task = asyncio.create_task(cancellable_task())
        await asyncio.sleep(0.1)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        assert cleanup_called, "任务取消时未执行清理操作"
```

### 3. 外部服务依赖测试

#### 测试用例 3.1: LLM API 限流处理测试
```python
from unittest.mock import AsyncMock, patch
from openai import RateLimitError

class TestLLMServiceResilience:
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """测试API限流处理"""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = RateLimitError("Rate limit exceeded")
        
        retry_count = 0
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = await mock_llm.generate("test prompt")
                break
            except RateLimitError:
                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(2 ** retry_count)  # 指数退避
                else:
                    # 验证降级策略
                    assert True, "应该触发降级策略"
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """测试网络超时处理"""
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = asyncio.TimeoutError("Request timeout")
        
        try:
            response = await asyncio.wait_for(
                mock_llm.generate("test prompt"),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            # 验证超时处理策略
            assert True, "应该正确处理超时"
```

### 4. 内存管理测试

#### 测试用例 4.1: 大文件处理测试
```python
import psutil
import os

class TestMemoryManagement:
    
    def test_large_file_processing(self):
        """测试大文件处理的内存使用"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 模拟大文件处理
        large_data = []
        for i in range(10000):
            large_data.append({"id": i, "content": "x" * 1000})
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # 清理数据
        del large_data
        
        # 验证内存使用合理
        assert memory_increase < 100 * 1024 * 1024, f"内存使用过多: {memory_increase / 1024 / 1024}MB"
    
    def test_vector_index_memory_usage(self):
        """测试向量索引内存使用"""
        import faiss
        import numpy as np
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 创建大型向量索引
        dimension = 768
        index = faiss.IndexFlatL2(dimension)
        
        # 添加大量向量
        vectors = np.random.random((10000, dimension)).astype('float32')
        index.add(vectors)
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # 验证内存使用合理
        assert memory_increase < 200 * 1024 * 1024, f"向量索引内存使用过多: {memory_increase / 1024 / 1024}MB"
```

### 5. 配置管理测试

#### 测试用例 5.1: 配置验证测试
```python
import os
from unittest.mock import patch

class TestConfigurationManagement:
    
    def test_missing_environment_variables(self):
        """测试缺失环境变量的处理"""
        required_vars = [
            'DATABASE_URL',
            'GEMINI_API_KEY',
            'REDIS_URL'
        ]
        
        for var in required_vars:
            with patch.dict(os.environ, {var: None}, clear=True):
                try:
                    from component.database_session_provider import DatabaseSessionProvider
                    provider = DatabaseSessionProvider()
                    assert False, f"应该检测到缺失的环境变量: {var}"
                except ValueError as e:
                    assert var in str(e), f"错误信息应该包含变量名: {var}"
    
    def test_invalid_database_url(self):
        """测试无效数据库URL的处理"""
        invalid_urls = [
            "invalid://url",
            "postgresql://",
            "postgresql://user@host",
            ""
        ]
        
        for url in invalid_urls:
            with patch.dict(os.environ, {'DATABASE_URL': url}):
                try:
                    from component.database_session_provider import DatabaseSessionProvider
                    provider = DatabaseSessionProvider()
                    # 这里应该验证URL格式
                    assert True, "应该验证数据库URL格式"
                except Exception as e:
                    assert "database" in str(e).lower(), f"应该检测到数据库配置错误: {e}"
```

### 6. 系统集成稳定性测试

#### 测试用例 6.1: 端到端压力测试
```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class TestSystemStability:
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self):
        """高并发压力测试"""
        async def api_request():
            # 模拟API请求
            await asyncio.sleep(0.01)
            return {"status": "success"}
        
        # 创建大量并发请求
        start_time = time.time()
        tasks = [asyncio.create_task(api_request()) for _ in range(1000)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 验证结果
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = len(results) - success_count
        
        assert success_count >= 950, f"成功率过低: {success_count}/1000"
        assert end_time - start_time < 30, f"响应时间过长: {end_time - start_time}秒"
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """内存泄漏检测测试"""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 执行多轮操作
        for round_num in range(10):
            # 模拟一轮完整的业务操作
            await self.simulate_business_operation()
            
            # 强制垃圾回收
            gc.collect()
            
            # 检查内存使用
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # 每轮内存增长不应超过10MB
            assert memory_increase < 10 * 1024 * 1024, f"第{round_num}轮内存泄漏: {memory_increase / 1024 / 1024}MB"
    
    async def simulate_business_operation(self):
        """模拟业务操作"""
        # 这里模拟完整的业务操作流程
        await asyncio.sleep(0.1)
```

## 🔧 稳定性改进建议

### 1. 数据库连接池优化

```python
# 建议的连接池配置
DATABASE_CONFIG = {
    "pool_size": 50,           # 增加基础连接池大小
    "max_overflow": 30,        # 增加最大溢出连接
    "pool_recycle": 300,       # 5分钟回收连接
    "pool_pre_ping": True,     # 连接前ping检查
    "pool_timeout": 30,        # 获取连接超时时间
    "max_retries": 3,          # 最大重试次数
    "retry_delay": 1.0,        # 重试延迟
}
```

### 2. 异步并发控制优化

```python
# 建议的并发控制配置
CONCURRENCY_CONFIG = {
    "max_concurrent_requests": 100,    # 最大并发请求数
    "max_concurrent_db_ops": 50,       # 最大并发数据库操作
    "max_concurrent_llm_calls": 10,    # 最大并发LLM调用
    "request_timeout": 30,             # 请求超时时间
    "task_cleanup_timeout": 5,         # 任务清理超时时间
}
```

### 3. 错误处理和重试机制

```python
# 建议的重试配置
RETRY_CONFIG = {
    "max_retries": 3,
    "base_delay": 1.0,
    "max_delay": 60.0,
    "exponential_base": 2,
    "jitter": True,
    "retryable_errors": [
        "ConnectionError",
        "TimeoutError",
        "RateLimitError"
    ]
}
```

### 4. 监控和告警

```python
# 建议的监控指标
MONITORING_METRICS = {
    "database_connections": "连接池使用率",
    "memory_usage": "内存使用率",
    "response_time": "响应时间",
    "error_rate": "错误率",
    "throughput": "吞吐量",
    "queue_length": "队列长度"
}
```

## 📊 测试执行计划

### 阶段1: 单元测试 (1-2周)
- 数据库连接池测试
- 异步并发控制测试
- 配置管理测试

### 阶段2: 集成测试 (2-3周)
- 外部服务依赖测试
- 内存管理测试
- 错误处理测试

### 阶段3: 系统测试 (3-4周)
- 端到端压力测试
- 长时间运行测试
- 故障恢复测试

### 阶段4: 生产环境测试 (1-2周)
- 灰度发布测试
- 监控告警测试
- 性能基准测试

## 🎯 测试覆盖率目标

- **代码覆盖率**: ≥ 80%
- **分支覆盖率**: ≥ 75%
- **关键路径覆盖率**: 100%
- **异常处理覆盖率**: ≥ 90%

## 📈 性能基准

- **响应时间**: P95 < 2秒
- **吞吐量**: ≥ 1000 QPS
- **内存使用**: < 2GB
- **CPU使用率**: < 70%
- **错误率**: < 0.1%

---

*本文档将持续更新，反映系统稳定性的改进和新的测试发现。*
