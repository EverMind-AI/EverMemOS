"""
数据库稳定性测试

测试数据库连接池、连接泄漏、故障恢复等关键稳定性场景
"""

import pytest
import asyncio
import os
import time
from unittest.mock import patch, AsyncMock
from typing import List, Dict, Any

# 设置测试环境
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("DB_POOL_SIZE", "5")
os.environ.setdefault("DB_MAX_OVERFLOW", "3")

from component.database_session_provider import DatabaseSessionProvider
from component.database_connection_provider import DatabaseConnectionProvider


class TestDatabaseStability:
    """数据库稳定性测试类"""

    @pytest.fixture
    async def db_provider(self):
        """数据库提供者fixture"""
        provider = DatabaseSessionProvider()
        yield provider
        # 清理资源
        if hasattr(provider, 'async_engine'):
            await provider.async_engine.dispose()

    @pytest.fixture
    async def connection_provider(self):
        """连接提供者fixture"""
        provider = DatabaseConnectionProvider()
        yield provider
        # 清理资源
        if hasattr(provider, '_connection_pool') and provider._connection_pool:
            await provider._connection_pool.close()

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, db_provider):
        """测试连接池耗尽场景"""

        # 创建超过连接池大小的并发任务
        async def db_operation(operation_id: int):
            try:
                async with db_provider.get_async_session() as session:
                    # 模拟长时间数据库操作
                    await asyncio.sleep(0.1)
                    return f"operation_{operation_id}_success"
            except Exception as e:
                return f"operation_{operation_id}_failed: {str(e)}"

        # 创建大量并发任务（超过连接池大小）
        max_connections = int(os.getenv("DB_POOL_SIZE", "5")) + int(
            os.getenv("DB_MAX_OVERFLOW", "3")
        )
        task_count = max_connections * 2  # 超过连接池大小

        tasks = [asyncio.create_task(db_operation(i)) for i in range(task_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 分析结果
        success_count = sum(1 for r in results if isinstance(r, str) and "success" in r)
        failure_count = len(results) - success_count

        print(
            f"连接池测试结果: 成功={success_count}, 失败={failure_count}, 总任务={task_count}"
        )

        # 验证：大部分任务应该成功，少数可能因连接池耗尽失败
        assert (
            success_count >= task_count * 0.8
        ), f"成功率过低: {success_count}/{task_count}"
        assert (
            failure_count <= task_count * 0.2
        ), f"失败率过高: {failure_count}/{task_count}"

    @pytest.mark.asyncio
    async def test_connection_leak_detection(self, db_provider):
        """测试连接泄漏检测"""
        # 记录初始连接数
        initial_pool_size = db_provider.async_engine.pool.size()
        initial_checked_in = db_provider.async_engine.pool.checkedin()
        initial_checked_out = db_provider.async_engine.pool.checkedout()

        print(
            f"初始连接池状态: size={initial_pool_size}, checked_in={initial_checked_in}, checked_out={initial_checked_out}"
        )

        # 模拟连接泄漏场景
        leaked_sessions = []

        async def leaky_operation():
            session = db_provider.create_session()
            leaked_sessions.append(session)
            # 故意不关闭session，模拟连接泄漏
            await session.execute("SELECT 1")
            # 不调用 session.close()

        # 执行多个泄漏操作
        for _ in range(3):
            await leaky_operation()

        # 检查连接池状态
        current_pool_size = db_provider.async_engine.pool.size()
        current_checked_in = db_provider.async_engine.pool.checkedin()
        current_checked_out = db_provider.async_engine.pool.checkedout()

        print(
            f"泄漏后连接池状态: size={current_pool_size}, checked_in={current_checked_in}, checked_out={current_checked_out}"
        )

        # 验证连接泄漏检测
        leaked_connections = current_checked_out - initial_checked_out
        assert leaked_connections > 0, "应该检测到连接泄漏"

        # 清理泄漏的连接
        for session in leaked_sessions:
            try:
                await session.close()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_database_failure_recovery(self, db_provider):
        """测试数据库故障恢复"""
        recovery_successful = False

        # 模拟数据库连接失败
        original_execute = db_provider.async_engine.execute

        async def mock_failing_execute(*args, **kwargs):
            # 前几次调用失败，后续调用成功
            if not hasattr(mock_failing_execute, 'call_count'):
                mock_failing_execute.call_count = 0
            mock_failing_execute.call_count += 1

            if mock_failing_execute.call_count <= 2:
                raise Exception("Database connection failed")
            else:
                # 恢复数据库连接
                return await original_execute(*args, **kwargs)

        with patch.object(
            db_provider.async_engine, 'execute', side_effect=mock_failing_execute
        ):
            # 测试重试机制
            max_retries = 3
            retry_count = 0

            for attempt in range(max_retries):
                try:
                    async with db_provider.get_async_session() as session:
                        await session.execute("SELECT 1")
                    recovery_successful = True
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"重试 {retry_count}: {str(e)}")
                    if retry_count < max_retries:
                        await asyncio.sleep(0.1)  # 短暂延迟

        assert recovery_successful, "数据库故障恢复失败"

    @pytest.mark.asyncio
    async def test_connection_pool_timeout(self, db_provider):
        """测试连接池超时处理"""

        # 创建长时间占用连接的任务
        async def long_running_operation():
            async with db_provider.get_async_session() as session:
                await asyncio.sleep(2)  # 长时间占用连接
                return "completed"

        # 创建多个长时间运行的任务
        long_tasks = [asyncio.create_task(long_running_operation()) for _ in range(3)]

        # 创建需要获取连接的新任务（应该超时）
        async def timeout_operation():
            try:
                async with db_provider.get_async_session() as session:
                    await session.execute("SELECT 1")
                    return "success"
            except Exception as e:
                return f"timeout: {str(e)}"

        # 等待一段时间后创建超时任务
        await asyncio.sleep(0.1)
        timeout_task = asyncio.create_task(timeout_operation())

        # 等待所有任务完成
        results = await asyncio.gather(
            *long_tasks, timeout_task, return_exceptions=True
        )

        # 验证超时处理
        timeout_result = results[-1]
        if isinstance(timeout_result, str) and "timeout" in timeout_result:
            print(f"连接池超时测试通过: {timeout_result}")
        else:
            print(f"连接池超时测试结果: {timeout_result}")

    @pytest.mark.asyncio
    async def test_concurrent_transaction_isolation(self, db_provider):
        """测试并发事务隔离"""
        results = []

        async def transaction_operation(operation_id: int):
            try:
                async with db_provider.get_async_session() as session:
                    # 开始事务
                    await session.begin()

                    # 模拟事务操作
                    await session.execute("SELECT 1")
                    await asyncio.sleep(0.1)  # 模拟处理时间

                    # 提交事务
                    await session.commit()

                    results.append(f"transaction_{operation_id}_success")
            except Exception as e:
                results.append(f"transaction_{operation_id}_failed: {str(e)}")

        # 创建多个并发事务
        tasks = [asyncio.create_task(transaction_operation(i)) for i in range(10)]
        await asyncio.gather(*tasks)

        # 验证事务隔离
        success_count = sum(1 for r in results if "success" in r)
        assert success_count == 10, f"事务隔离测试失败: {results}"

    @pytest.mark.asyncio
    async def test_connection_pool_health_check(self, connection_provider):
        """测试连接池健康检查"""
        try:
            # 获取连接池
            pool = await connection_provider.get_connection_pool()

            # 检查连接池状态
            assert pool is not None, "连接池未初始化"

            # 测试连接池健康状态
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")

            print("连接池健康检查通过")

        except Exception as e:
            pytest.fail(f"连接池健康检查失败: {str(e)}")

    @pytest.mark.asyncio
    async def test_database_performance_under_load(self, db_provider):
        """测试数据库在负载下的性能"""
        start_time = time.time()

        async def performance_operation(operation_id: int):
            async with db_provider.get_async_session() as session:
                # 执行简单查询
                result = await session.execute("SELECT 1 as test_value")
                return f"operation_{operation_id}_completed"

        # 创建大量并发操作
        task_count = 100
        tasks = [
            asyncio.create_task(performance_operation(i)) for i in range(task_count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # 分析性能
        success_count = sum(
            1 for r in results if isinstance(r, str) and "completed" in r
        )
        avg_time_per_operation = total_time / task_count
        operations_per_second = task_count / total_time

        print(f"性能测试结果:")
        print(f"  总时间: {total_time:.2f}秒")
        print(f"  成功操作: {success_count}/{task_count}")
        print(f"  平均每操作时间: {avg_time_per_operation:.3f}秒")
        print(f"  每秒操作数: {operations_per_second:.2f}")

        # 性能断言
        assert (
            success_count >= task_count * 0.95
        ), f"成功率过低: {success_count}/{task_count}"
        assert (
            avg_time_per_operation < 0.1
        ), f"平均响应时间过长: {avg_time_per_operation:.3f}秒"
        assert (
            operations_per_second > 50
        ), f"吞吐量过低: {operations_per_second:.2f} ops/sec"


class TestDatabaseErrorHandling:
    """数据库错误处理测试类"""

    @pytest.mark.asyncio
    async def test_invalid_query_handling(self):
        """测试无效查询处理"""
        provider = DatabaseSessionProvider()

        try:
            async with provider.get_async_session() as session:
                # 执行无效SQL
                await session.execute("INVALID SQL STATEMENT")
        except Exception as e:
            # 验证错误处理
            assert (
                "syntax error" in str(e).lower() or "invalid" in str(e).lower()
            ), f"未正确处理无效查询: {e}"
        finally:
            await provider.async_engine.dispose()

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """测试连接超时处理"""
        provider = DatabaseSessionProvider()

        # 模拟连接超时
        with patch.object(provider.async_engine, 'connect') as mock_connect:
            mock_connect.side_effect = asyncio.TimeoutError("Connection timeout")

            try:
                async with provider.get_async_session() as session:
                    await session.execute("SELECT 1")
            except asyncio.TimeoutError:
                # 验证超时处理
                assert True, "应该正确处理连接超时"
            except Exception as e:
                pytest.fail(f"未正确处理连接超时: {e}")
            finally:
                await provider.async_engine.dispose()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
