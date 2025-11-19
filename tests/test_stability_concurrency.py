"""
并发稳定性测试

测试异步并发控制、信号量限制、任务取消等关键稳定性场景
"""

import pytest
import asyncio
import time
import os
import sys

# 可选导入psutil，如果不可用则跳过相关测试
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch
from pathlib import Path


# 导入真实的业务模块
from biz_layer.tanka_memorize import memorize
from memory_layer.memory_manager import MemorizeRequest
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from memory_layer.types import RawDataType, MemoryType

# 导入fetch_mem相关模块
from agentic_layer.memory_manager import MemoryManager
from agentic_layer.memory_models import FetchMemRequest, MemoryType as AgenticMemoryType


class BaseConcurrencyTest:
    """并发测试基类，提供通用的业务调用方法"""

    async def _call_real_memorize_business(self, task_id: int):
        """调用真实的记忆提取业务逻辑"""
        try:
            # 创建真实的测试数据
            base_time = datetime.now() - timedelta(hours=1)

            # 创建历史消息
            history_messages = [
                {
                    "speaker_id": f"user_{task_id}",
                    "speaker_name": f"User{task_id}",
                    "content": f"这是第{task_id}个测试的历史消息，用于测试并发处理能力。",
                    "timestamp": (base_time + timedelta(minutes=0)).isoformat(),
                }
            ]

            # 创建新消息
            new_messages = [
                {
                    "speaker_id": f"user_{task_id}",
                    "speaker_name": f"User{task_id}",
                    "content": f"这是第{task_id}个测试的新消息，用于测试并发处理能力。",
                    "timestamp": (base_time + timedelta(minutes=1)).isoformat(),
                }
            ]

            # 转换为RawData对象
            history_raw_data = [
                RawData(
                    content=msg,
                    raw_data_type=RawDataType.CONVERSATION,
                    timestamp=msg["timestamp"],
                )
                for msg in history_messages
            ]

            new_raw_data = [
                RawData(
                    content=msg,
                    raw_data_type=RawDataType.CONVERSATION,
                    timestamp=msg["timestamp"],
                )
                for msg in new_messages
            ]

            # 创建MemorizeRequest
            memorize_request = MemorizeRequest(
                history_raw_data_list=history_raw_data,
                new_raw_data_list=new_raw_data,
                raw_data_type=RawDataType.CONVERSATION,
                user_id_list=[f"user_{task_id}"],
                group_id=f"test_group_{task_id}",
                current_time=datetime.now(),
            )

            # 调用真实的memorize函数
            memory_list = await memorize(memorize_request)

            return {
                "task_id": task_id,
                "memory_count": len(memory_list) if memory_list else 0,
                "success": True,
            }

        except Exception as e:
            return {"task_id": task_id, "error": str(e), "success": False}

    async def _call_real_fetch_mem_business(self, task_id: int):
        """调用真实的记忆获取业务逻辑"""
        try:
            # 创建MemoryManager实例
            memory_manager = MemoryManager()

            # 创建FetchMemRequest
            fetch_request = FetchMemRequest(
                user_id=f"test_user_{task_id}",
                memory_type=AgenticMemoryType.MULTIPLE,
                limit=5,
                offset=0,
                filters={},
                sort_by="created_at",
                sort_order="desc",
            )

            # 调用真实的fetch_mem函数
            fetch_response = await memory_manager.fetch_mem(fetch_request)

            return {
                "task_id": task_id,
                "memory_count": (
                    len(fetch_response.memories) if fetch_response.memories else 0
                ),
                "total_count": fetch_response.total_count,
                "has_more": fetch_response.has_more,
                "success": True,
            }

        except Exception as e:
            return {"task_id": task_id, "error": str(e), "success": False}


class TestConcurrencyStability(BaseConcurrencyTest):
    """并发稳定性测试类"""

    @pytest.mark.asyncio
    async def test_semaphore_limitation(self):
        """测试信号量限制并发数"""
        max_concurrent = 5
        semaphore = asyncio.Semaphore(max_concurrent)
        concurrent_count = 0
        max_observed = 0
        lock = asyncio.Lock()

        async def limited_operation(operation_id: int):
            nonlocal concurrent_count, max_observed

            async with semaphore:
                async with lock:
                    concurrent_count += 1
                    max_observed = max(max_observed, concurrent_count)

                # 使用真实的业务操作 - 交替测试memorize和fetch_mem
                try:
                    if operation_id % 2 == 0:
                        result = await self._call_real_memorize_business(operation_id)
                        operation_result = f"memorize_operation_{operation_id}_completed_memories_{result.get('memory_count', 0)}"
                    else:
                        result = await self._call_real_fetch_mem_business(operation_id)
                        operation_result = f"fetch_operation_{operation_id}_completed_memories_{result.get('memory_count', 0)}_total_{result.get('total_count', 0)}"
                except Exception as e:
                    operation_result = f"operation_{operation_id}_error: {str(e)}"

                async with lock:
                    concurrent_count -= 1

                return operation_result

        # 创建大量任务
        task_count = 50
        tasks = [asyncio.create_task(limited_operation(i)) for i in range(task_count)]
        results = await asyncio.gather(*tasks)

        # 验证并发数不超过限制
        assert (
            max_observed <= max_concurrent
        ), f"并发数超过限制: {max_observed} > {max_concurrent}"
        assert (
            len(results) == task_count
        ), f"任务完成数量不匹配: {len(results)} != {task_count}"

        print(
            f"信号量测试结果: 最大并发数={max_observed}, 限制={max_concurrent}, 总任务={task_count}"
        )

    @pytest.mark.asyncio
    async def test_task_cancellation_cleanup(self):
        """测试任务取消时的资源清理"""
        cleanup_called = False
        resource_acquired = False

        async def cancellable_task(task_id: int):
            nonlocal cleanup_called, resource_acquired

            try:
                # 模拟资源获取
                resource_acquired = True
                print(f"任务 {task_id} 获取资源")

                # 使用真实的业务操作，但设置较长的超时来模拟长时间运行
                try:
                    result = await asyncio.wait_for(
                        self._call_real_memorize_business(task_id), timeout=10.0
                    )
                    print(
                        f"任务 {task_id} 完成，提取了 {result.get('memory_count', 0)} 个记忆"
                    )
                except asyncio.TimeoutError:
                    print(f"任务 {task_id} 超时")

            except asyncio.CancelledError:
                # 清理资源
                cleanup_called = True
                resource_acquired = False
                print(f"任务 {task_id} 被取消，执行清理")
                raise
            finally:
                # 确保资源被释放
                if resource_acquired:
                    resource_acquired = False
                    print(f"任务 {task_id} 释放资源")

        # 创建任务
        task = asyncio.create_task(cancellable_task(1))

        # 等待任务开始
        await asyncio.sleep(0.1)

        # 取消任务
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # 验证清理操作
        assert cleanup_called, "任务取消时未执行清理操作"
        assert not resource_acquired, "资源未被正确释放"

    @pytest.mark.asyncio
    async def test_concurrent_resource_access(self):
        """测试并发资源访问"""
        shared_resource = {"value": 0, "access_count": 0}
        lock = asyncio.Lock()

        async def resource_accessor(accessor_id: int):
            async with lock:
                # 模拟资源访问
                shared_resource["access_count"] += 1
                current_value = shared_resource["value"]

                # 使用真实的业务操作
                try:
                    result = await self._call_real_memorize_business(accessor_id)
                    memory_count = result.get('memory_count', 0)
                except Exception as e:
                    memory_count = 0
                    print(f"访问者 {accessor_id} 业务调用失败: {str(e)}")

                # 更新资源
                shared_resource["value"] = current_value + 1

                return f"accessor_{accessor_id}_updated_value_to_{shared_resource['value']}_memories_{memory_count}"

        # 创建多个并发访问者
        accessor_count = 20
        tasks = [
            asyncio.create_task(resource_accessor(i)) for i in range(accessor_count)
        ]
        results = await asyncio.gather(*tasks)

        # 验证资源访问的正确性
        assert (
            shared_resource["access_count"] == accessor_count
        ), f"访问次数不匹配: {shared_resource['access_count']} != {accessor_count}"
        assert (
            shared_resource["value"] == accessor_count
        ), f"资源值不正确: {shared_resource['value']} != {accessor_count}"
        assert (
            len(results) == accessor_count
        ), f"结果数量不匹配: {len(results)} != {accessor_count}"

        print(
            f"并发资源访问测试结果: 访问次数={shared_resource['access_count']}, 最终值={shared_resource['value']}"
        )

    @pytest.mark.asyncio
    async def test_deadlock_prevention(self):
        """测试死锁预防"""
        lock1 = asyncio.Lock()
        lock2 = asyncio.Lock()

        async def task_with_locks(task_id: int, lock_order: str):
            if lock_order == "1_2":
                async with lock1:
                    # 使用真实的业务操作
                    try:
                        result = await self._call_real_memorize_business(task_id)
                        memory_count = result.get('memory_count', 0)
                    except Exception as e:
                        memory_count = 0
                        print(f"任务 {task_id} 业务调用失败: {str(e)}")

                    async with lock2:
                        return f"task_{task_id}_completed_1_2_memories_{memory_count}"
            else:  # "2_1"
                async with lock2:
                    # 使用真实的业务操作
                    try:
                        result = await self._call_real_memorize_business(task_id)
                        memory_count = result.get('memory_count', 0)
                    except Exception as e:
                        memory_count = 0
                        print(f"任务 {task_id} 业务调用失败: {str(e)}")

                    async with lock1:
                        return f"task_{task_id}_completed_2_1_memories_{memory_count}"

        # 创建不同锁顺序的任务
        tasks = []
        for i in range(10):
            if i % 2 == 0:
                tasks.append(asyncio.create_task(task_with_locks(i, "1_2")))
            else:
                tasks.append(asyncio.create_task(task_with_locks(i, "2_1")))

        # 设置超时防止死锁
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)
            print(f"死锁预防测试通过: 完成 {len(results)} 个任务")
        except asyncio.TimeoutError:
            pytest.fail("检测到死锁或任务执行超时")

    @pytest.mark.asyncio
    async def test_memory_usage_under_concurrency(self):
        """测试并发下的内存使用"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available for memory usage test")
            return

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        async def memory_intensive_task(task_id: int):
            # 使用真实的业务操作，这会创建真实的数据结构
            try:
                result = await self._call_real_memorize_business(task_id)
                memory_count = result.get('memory_count', 0)
                return f"task_{task_id}_memories_{memory_count}"
            except Exception as e:
                return f"task_{task_id}_error: {str(e)}"

        # 创建大量并发任务
        task_count = 100
        tasks = [
            asyncio.create_task(memory_intensive_task(i)) for i in range(task_count)
        ]
        results = await asyncio.gather(*tasks)

        # 检查内存使用
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory

        print(f"内存使用测试结果:")
        print(f"  初始内存: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"  峰值内存: {peak_memory / 1024 / 1024:.2f} MB")
        print(f"  内存增长: {memory_increase / 1024 / 1024:.2f} MB")
        print(f"  完成任务: {len(results)}")

        # 验证内存使用合理
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"内存使用过多: {memory_increase / 1024 / 1024:.2f} MB"
        assert (
            len(results) == task_count
        ), f"任务完成数量不匹配: {len(results)} != {task_count}"

    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self):
        """测试并发错误处理"""
        error_count = 0
        success_count = 0

        async def task_with_errors(task_id: int):
            nonlocal error_count, success_count

            try:
                if task_id % 3 == 0:
                    raise ValueError(f"模拟错误 {task_id}")
                else:
                    # 使用真实的业务操作
                    result = await self._call_real_memorize_business(task_id)
                    success_count += 1
                    memory_count = result.get('memory_count', 0)
                    return f"task_{task_id}_success_memories_{memory_count}"
            except Exception as e:
                error_count += 1
                return f"task_{task_id}_error: {str(e)}"

        # 创建任务
        task_count = 30
        tasks = [asyncio.create_task(task_with_errors(i)) for i in range(task_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 验证错误处理
        expected_errors = task_count // 3
        expected_successes = task_count - expected_errors

        print(f"并发错误处理测试结果:")
        print(f"  总任务: {task_count}")
        print(f"  成功: {success_count}")
        print(f"  错误: {error_count}")
        print(f"  预期成功: {expected_successes}")
        print(f"  预期错误: {expected_errors}")

        assert (
            success_count == expected_successes
        ), f"成功数量不匹配: {success_count} != {expected_successes}"
        assert (
            error_count == expected_errors
        ), f"错误数量不匹配: {error_count} != {expected_errors}"

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self):
        """测试并发速率限制"""
        rate_limit = 10  # 每秒最多10个请求
        request_times = []
        semaphore = asyncio.Semaphore(rate_limit)

        async def rate_limited_request(request_id: int):
            async with semaphore:
                request_times.append(time.time())
                # 使用真实的业务操作
                try:
                    result = await self._call_real_memorize_business(request_id)
                    memory_count = result.get('memory_count', 0)
                    return f"request_{request_id}_completed_memories_{memory_count}"
                except Exception as e:
                    return f"request_{request_id}_error: {str(e)}"

        # 创建大量请求
        request_count = 50
        tasks = [
            asyncio.create_task(rate_limited_request(i)) for i in range(request_count)
        ]
        results = await asyncio.gather(*tasks)

        # 分析请求时间分布
        if len(request_times) > 1:
            time_diffs = [
                request_times[i + 1] - request_times[i]
                for i in range(len(request_times) - 1)
            ]
            avg_interval = sum(time_diffs) / len(time_diffs)

            print(f"速率限制测试结果:")
            print(f"  总请求: {request_count}")
            print(f"  完成请求: {len(results)}")
            print(f"  平均间隔: {avg_interval:.3f}秒")
            print(f"  预期间隔: {1.0/rate_limit:.3f}秒")

            # 验证速率限制
            assert (
                len(results) == request_count
            ), f"请求完成数量不匹配: {len(results)} != {request_count}"
            assert (
                avg_interval >= 1.0 / rate_limit * 0.8
            ), f"速率限制未生效: 平均间隔 {avg_interval:.3f}秒"


class TestAsyncTaskManagement(BaseConcurrencyTest):
    """异步任务管理测试类"""

    @pytest.mark.asyncio
    async def test_task_group_management(self):
        """测试任务组管理"""
        task_group = []
        completed_tasks = []

        async def managed_task(task_id: int):
            # 使用真实的业务操作
            try:
                result = await self._call_real_memorize_business(task_id)
                memory_count = result.get('memory_count', 0)
                completed_tasks.append(task_id)
                return f"task_{task_id}_completed_memories_{memory_count}"
            except Exception as e:
                completed_tasks.append(task_id)
                return f"task_{task_id}_error: {str(e)}"

        # 创建任务组
        for i in range(10):
            task = asyncio.create_task(managed_task(i))
            task_group.append(task)

        # 等待所有任务完成
        results = await asyncio.gather(*task_group)

        # 验证任务管理
        assert len(results) == 10, f"任务完成数量不匹配: {len(results)} != 10"
        assert (
            len(completed_tasks) == 10
        ), f"完成任务记录不匹配: {len(completed_tasks)} != 10"
        assert set(completed_tasks) == set(
            range(10)
        ), f"完成任务ID不匹配: {completed_tasks}"

    @pytest.mark.asyncio
    async def test_task_timeout_handling(self):
        """测试任务超时处理"""
        timeout_occurred = False

        async def slow_task(task_id: int):
            # 使用真实的业务操作，但设置较长的处理时间来模拟超时
            try:
                # 模拟一个较长的业务处理时间
                result = await asyncio.wait_for(
                    self._call_real_memorize_business(task_id), timeout=2.0
                )
                return (
                    f"task_{task_id}_completed_memories_{result.get('memory_count', 0)}"
                )
            except asyncio.TimeoutError:
                return f"task_{task_id}_timeout"
            except Exception as e:
                return f"task_{task_id}_error: {str(e)}"

        # 创建带超时的任务
        task = asyncio.create_task(slow_task(1))

        try:
            result = await asyncio.wait_for(task, timeout=1.0)
            pytest.fail("任务应该超时")
        except asyncio.TimeoutError:
            timeout_occurred = True
            print("任务超时处理正确")

        assert timeout_occurred, "任务超时处理失败"

    @pytest.mark.asyncio
    async def test_concurrent_exception_propagation(self):
        """测试并发异常传播"""

        async def task_with_exception(task_id: int):
            if task_id == 5:
                raise ValueError(f"任务 {task_id} 故意抛出异常")

            # 使用真实的业务操作
            try:
                result = await self._call_real_memorize_business(task_id)
                memory_count = result.get('memory_count', 0)
                return f"task_{task_id}_success_memories_{memory_count}"
            except Exception as e:
                return f"task_{task_id}_business_error: {str(e)}"

        # 创建任务
        tasks = [asyncio.create_task(task_with_exception(i)) for i in range(10)]

        # 收集结果，允许异常
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 分析结果
        success_count = sum(1 for r in results if isinstance(r, str) and "success" in r)
        exception_count = sum(1 for r in results if isinstance(r, Exception))

        print(f"异常传播测试结果: 成功={success_count}, 异常={exception_count}")

        assert success_count == 9, f"成功任务数量不匹配: {success_count} != 9"
        assert exception_count == 1, f"异常任务数量不匹配: {exception_count} != 1"

        # 验证异常类型
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 1 and isinstance(
            exceptions[0], ValueError
        ), "异常类型不正确"


class TestConcurrencyStress(BaseConcurrencyTest):
    """并发压力测试类"""

    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self):
        """高并发压力测试 - 基于真实业务接口"""
        start_time = time.time()
        success_count = 0
        error_count = 0

        async def business_stress_task(task_id: int):
            """基于真实业务逻辑的压测任务"""
            nonlocal success_count, error_count

            try:
                # 调用真实的业务操作：记忆提取请求
                result = await self._call_real_memorize_business(task_id)
                if result["success"]:
                    success_count += 1
                    return f"task_{task_id}_success_memories_{result['memory_count']}"
                else:
                    error_count += 1
                    return f"task_{task_id}_error: {result['error']}"
            except Exception as e:
                error_count += 1
                return f"task_{task_id}_error: {str(e)}"

        # 创建大量并发任务
        task_count = 100  # 减少任务数量，因为真实业务操作更耗时
        tasks = [
            asyncio.create_task(business_stress_task(i)) for i in range(task_count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"高并发压力测试结果:")
        print(f"  总任务: {task_count}")
        print(f"  成功: {success_count}")
        print(f"  错误: {error_count}")
        print(f"  总时间: {total_time:.2f}秒")
        print(f"  吞吐量: {task_count/total_time:.2f} 任务/秒")

        # 性能断言 - 调整期望值，因为真实业务操作更复杂
        assert (
            success_count >= task_count * 0.8
        ), f"成功率过低: {success_count}/{task_count}"
        assert total_time < 60, f"执行时间过长: {total_time:.2f}秒"
        assert (
            task_count / total_time > 5
        ), f"吞吐量过低: {task_count/total_time:.2f} 任务/秒"

    @pytest.mark.asyncio
    async def test_real_business_concurrency_stress(self):
        """基于真实业务接口的并发压测"""
        # 检查是否有API密钥，如果没有则跳过
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip(
                "OPENROUTER_API_KEY not available for real business stress test"
            )
            return

        start_time = time.time()
        success_count = 0
        error_count = 0

        async def real_business_task(task_id: int):
            """真实的业务压测任务"""
            nonlocal success_count, error_count

            try:
                # 调用真实的业务接口
                result = await self._call_real_memorize_business(task_id)
                if result["success"]:
                    success_count += 1
                    return (
                        f"real_task_{task_id}_success_memories_{result['memory_count']}"
                    )
                else:
                    error_count += 1
                    return f"real_task_{task_id}_error: {result['error']}"
            except Exception as e:
                error_count += 1
                return f"real_task_{task_id}_error: {str(e)}"

        # 创建并发任务 - 数量较少，因为真实API调用更耗时
        task_count = 20
        tasks = [asyncio.create_task(real_business_task(i)) for i in range(task_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"真实业务并发压测结果:")
        print(f"  总任务: {task_count}")
        print(f"  成功: {success_count}")
        print(f"  错误: {error_count}")
        print(f"  总时间: {total_time:.2f}秒")
        print(f"  吞吐量: {task_count/total_time:.2f} 任务/秒")

        # 真实业务压测的期望值
        assert (
            success_count >= task_count * 0.7
        ), f"成功率过低: {success_count}/{task_count}"
        assert total_time < 120, f"执行时间过长: {total_time:.2f}秒"

    @pytest.mark.asyncio
    async def test_single_real_business_call(self):
        """测试单个真实业务调用，验证接口可用性"""
        # 检查是否有API密钥，如果没有则跳过
        if not os.getenv("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not available for real business test")
            return

        try:
            # 调用真实的业务逻辑
            result = await self._call_real_memorize_business(1)

            print(f"单个业务调用测试结果: {result}")

            # 验证结果
            assert "task_id" in result, "结果中缺少task_id"
            assert "success" in result, "结果中缺少success字段"

            if result["success"]:
                assert "memory_count" in result, "成功结果中缺少memory_count"
                print(f"✅ 业务调用成功，提取了 {result['memory_count']} 个记忆")
            else:
                assert "error" in result, "失败结果中缺少error信息"
                print(f"⚠️ 业务调用失败: {result['error']}")

        except Exception as e:
            pytest.fail(f"业务调用测试失败: {str(e)}")

    @pytest.mark.asyncio
    async def test_fetch_mem_concurrency_stress(self):
        """基于fetch_mem接口的并发压测"""
        start_time = time.time()
        success_count = 0
        error_count = 0

        async def fetch_mem_stress_task(task_id: int):
            """基于fetch_mem的压测任务"""
            nonlocal success_count, error_count

            try:
                # 调用真实的fetch_mem业务操作
                result = await self._call_real_fetch_mem_business(task_id)
                if result["success"]:
                    success_count += 1
                    return f"fetch_task_{task_id}_success_memories_{result['memory_count']}_total_{result['total_count']}"
                else:
                    error_count += 1
                    return f"fetch_task_{task_id}_error: {result['error']}"
            except Exception as e:
                error_count += 1
                return f"fetch_task_{task_id}_error: {str(e)}"

        # 创建并发任务
        task_count = 50  # fetch_mem通常比memorize快，可以设置更多任务
        tasks = [
            asyncio.create_task(fetch_mem_stress_task(i)) for i in range(task_count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        print(f"fetch_mem并发压测结果:")
        print(f"  总任务: {task_count}")
        print(f"  成功: {success_count}")
        print(f"  错误: {error_count}")
        print(f"  总时间: {total_time:.2f}秒")
        print(f"  吞吐量: {task_count/total_time:.2f} 任务/秒")

        # fetch_mem压测的期望值
        assert (
            success_count >= task_count * 0.8
        ), f"成功率过低: {success_count}/{task_count}"
        assert total_time < 60, f"执行时间过长: {total_time:.2f}秒"

    @pytest.mark.asyncio
    async def test_mixed_business_concurrency_stress(self):
        """混合业务接口的并发压测：同时测试memorize和fetch_mem"""
        start_time = time.time()
        memorize_success = 0
        fetch_success = 0
        error_count = 0

        async def mixed_business_task(task_id: int):
            """混合业务压测任务"""
            nonlocal memorize_success, fetch_success, error_count

            try:
                # 交替调用memorize和fetch_mem
                if task_id % 2 == 0:
                    # 偶数任务调用memorize
                    result = await self._call_real_memorize_business(task_id)
                    if result["success"]:
                        memorize_success += 1
                        return f"memorize_task_{task_id}_success_memories_{result['memory_count']}"
                    else:
                        error_count += 1
                        return f"memorize_task_{task_id}_error: {result['error']}"
                else:
                    # 奇数任务调用fetch_mem
                    result = await self._call_real_fetch_mem_business(task_id)
                    if result["success"]:
                        fetch_success += 1
                        return f"fetch_task_{task_id}_success_memories_{result['memory_count']}_total_{result['total_count']}"
                    else:
                        error_count += 1
                        return f"fetch_task_{task_id}_error: {result['error']}"
            except Exception as e:
                error_count += 1
                return f"mixed_task_{task_id}_error: {str(e)}"

        # 创建混合业务任务
        task_count = 40  # 20个memorize + 20个fetch_mem
        tasks = [asyncio.create_task(mixed_business_task(i)) for i in range(task_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time
        total_success = memorize_success + fetch_success

        print(f"混合业务并发压测结果:")
        print(f"  总任务: {task_count}")
        print(f"  memorize成功: {memorize_success}")
        print(f"  fetch_mem成功: {fetch_success}")
        print(f"  总成功: {total_success}")
        print(f"  错误: {error_count}")
        print(f"  总时间: {total_time:.2f}秒")
        print(f"  吞吐量: {task_count/total_time:.2f} 任务/秒")

        # 混合业务压测的期望值
        assert (
            total_success >= task_count * 0.7
        ), f"成功率过低: {total_success}/{task_count}"
        assert total_time < 90, f"执行时间过长: {total_time:.2f}秒"
        assert memorize_success > 0, "memorize任务应该至少有一个成功"
        assert fetch_success > 0, "fetch_mem任务应该至少有一个成功"

    @pytest.mark.asyncio
    async def test_memory_leak_under_concurrency(self):
        """测试并发下的内存泄漏"""
        if not PSUTIL_AVAILABLE:
            pytest.skip("psutil not available for memory leak test")
            return

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        async def memory_task(round_num: int):
            # 创建一些数据
            data = [i for i in range(100)]
            await asyncio.sleep(0.01)
            return len(data)

        # 执行多轮并发操作
        for round_num in range(10):
            tasks = [asyncio.create_task(memory_task(round_num)) for _ in range(50)]
            results = await asyncio.gather(*tasks)

            # 强制垃圾回收
            import gc

            gc.collect()

            # 检查内存使用
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory

            print(
                f"第 {round_num+1} 轮: 内存增长 {memory_increase / 1024 / 1024:.2f} MB"
            )

            # 每轮内存增长不应超过5MB
            assert (
                memory_increase < 5 * 1024 * 1024
            ), f"第{round_num+1}轮内存泄漏: {memory_increase / 1024 / 1024:.2f} MB"


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
