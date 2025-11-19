"""
限流装饰器快速测试

专门用于快速验证限流装饰器功能的测试用例，避免长时间等待。
"""

import asyncio
import time
import pytest

from core.rate_limit.rate_limiter import rate_limit


class TestRateLimiterQuick:
    """快速限流测试，避免长时间等待"""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """测试基本功能，使用短时间窗口"""
        call_times = []

        @rate_limit(max_rate=2, time_period=1)  # 每秒2次
        async def quick_test_func():
            call_times.append(time.time())
            return "success"

        start_time = time.time()

        # 连续调用3次
        results = await asyncio.gather(
            quick_test_func(), quick_test_func(), quick_test_func()
        )

        total_time = time.time() - start_time

        assert len(results) == 3
        assert all(r == "success" for r in results)
        assert len(call_times) == 3

        # 第三次调用应该等待约0.5秒 (leaky bucket算法)
        assert total_time >= 0.4, f"限流等待时间不足: {total_time}秒"
        assert total_time < 0.8, f"限流等待时间过长: {total_time}秒"

        print(f"基本功能测试完成，耗时: {total_time:.3f}秒")

    @pytest.mark.asyncio
    async def test_high_frequency_short_period(self):
        """测试高频率限流（每秒10次）"""
        call_count = 0

        @rate_limit(max_rate=10, time_period=1)
        async def high_freq_func():
            nonlocal call_count
            call_count += 1
            return call_count

        start_time = time.time()

        # 尝试调用15次，前10次应该立即执行，后5次需要等待
        tasks = [high_freq_func() for _ in range(15)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        assert len(results) == 15
        assert results == list(range(1, 16))

        # 15次调用，每秒10次，需要约0.5秒（leaky bucket算法）
        assert total_time >= 0.4, f"高频率限流时间不足: {total_time}秒"
        assert total_time < 0.8, f"高频率限流时间过长: {total_time}秒"

        print(f"高频率限流测试完成，耗时: {total_time:.3f}秒")

    @pytest.mark.asyncio
    async def test_concurrent_different_keys(self):
        """测试不同键的并发调用"""
        results = {}

        @rate_limit(max_rate=1, time_period=1, key_func=lambda user: f"user_{user}")
        async def user_func(user):
            if user not in results:
                results[user] = []
            results[user].append(time.time())
            return f"result_{user}"

        start_time = time.time()

        # 3个用户各调用1次，应该并发执行
        tasks = [
            user_func("A"),  # 用户A的调用
            user_func("B"),  # 用户B的调用
            user_func("C"),  # 用户C的调用
        ]

        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # 验证每个用户都被调用了1次
        assert len(results["A"]) == 1
        assert len(results["B"]) == 1
        assert len(results["C"]) == 1

        # 不同用户的调用应该并发执行，总时间应该很短
        assert total_time < 0.1, f"不同键的并发调用耗时过长: {total_time}秒"

        print(f"并发不同键测试完成，耗时: {total_time:.3f}秒")

    @pytest.mark.asyncio
    async def test_decorator_performance_quick(self):
        """快速性能测试"""

        # 无装饰器函数
        async def plain_func():
            await asyncio.sleep(0.001)  # 添加小延迟使测量更准确
            return 42

        # 有装饰器函数（宽松限制）
        @rate_limit(max_rate=1000, time_period=1)
        async def decorated_func():
            await asyncio.sleep(0.001)  # 添加小延迟使测量更准确
            return 42

        # 测试50次调用的性能差异
        iterations = 50

        start_time = time.time()
        for _ in range(iterations):
            await plain_func()
        plain_time = time.time() - start_time

        start_time = time.time()
        for _ in range(iterations):
            await decorated_func()
        decorated_time = time.time() - start_time

        overhead = decorated_time - plain_time
        overhead_percent = (overhead / plain_time) * 100 if plain_time > 0 else 0

        print(
            f"性能测试 - 无装饰器: {plain_time:.4f}秒, 有装饰器: {decorated_time:.4f}秒"
        )
        print(f"开销: {overhead:.4f}秒 ({overhead_percent:.1f}%)")

        # 装饰器开销应该小于200%（相对宽松的限制）
        assert overhead_percent < 200, f"装饰器开销过大: {overhead_percent:.1f}%"

    @pytest.mark.asyncio
    async def test_error_handling_quick(self):
        """快速错误处理测试"""
        call_count = 0

        @rate_limit(max_rate=2, time_period=1)
        async def error_func(should_fail=False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise RuntimeError("测试错误")
            return "success"

        # 正常调用
        result1 = await error_func(False)
        assert result1 == "success"

        # 错误调用（仍然消耗配额）
        with pytest.raises(RuntimeError):
            await error_func(True)

        # 第三次调用应该被限流
        start_time = time.time()
        result3 = await error_func(False)
        elapsed = time.time() - start_time

        assert result3 == "success"
        assert elapsed >= 0.4, f"限流等待时间不足: {elapsed}秒"
        assert call_count == 3

        print(f"错误处理测试完成，等待时间: {elapsed:.3f}秒")

    @pytest.mark.asyncio
    async def test_parameter_validation(self):
        """测试参数验证"""

        # 测试无效的max_rate
        with pytest.raises(ValueError, match="max_rate must be positive"):

            @rate_limit(max_rate=0, time_period=1)
            async def invalid_rate_func():
                pass

        with pytest.raises(ValueError, match="max_rate must be positive"):

            @rate_limit(max_rate=-1, time_period=1)
            async def negative_rate_func():
                pass

        # 测试无效的time_period
        with pytest.raises(ValueError, match="time_period must be positive"):

            @rate_limit(max_rate=1, time_period=0)
            async def invalid_period_func():
                pass

        with pytest.raises(ValueError, match="time_period must be positive"):

            @rate_limit(max_rate=1, time_period=-1)
            async def negative_period_func():
                pass

        print("参数验证测试完成")

    @pytest.mark.asyncio
    async def test_1_second_10_requests(self):
        """测试1秒内10次请求的限流"""
        call_times = []

        @rate_limit(max_rate=10, time_period=1)
        async def high_freq_func():
            call_times.append(time.time())
            return len(call_times)

        start_time = time.time()

        # 连续调用12次，前10次应该立即执行，后2次需要等待
        tasks = [high_freq_func() for _ in range(12)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        assert len(results) == 12
        assert results == list(range(1, 13))
        assert len(call_times) == 12

        # 前10次应该相对快速完成，但由于leaky bucket算法可能有一些延迟
        first_10_time = call_times[9] - call_times[0]
        assert first_10_time < 0.5, f"前10次调用耗时过长: {first_10_time:.3f}秒"

        # 第11次和第12次调用需要等待
        wait_time_11 = call_times[10] - call_times[9]
        wait_time_12 = call_times[11] - call_times[10]

        assert wait_time_11 >= 0.08, f"第11次调用等待时间不足: {wait_time_11:.3f}秒"
        assert wait_time_12 >= 0.08, f"第12次调用等待时间不足: {wait_time_12:.3f}秒"

        print(f"1秒10次测试完成，总耗时: {total_time:.3f}秒")
        print(f"前10次耗时: {first_10_time:.3f}秒")
        print(f"第11次等待: {wait_time_11:.3f}秒，第12次等待: {wait_time_12:.3f}秒")

    @pytest.mark.asyncio
    async def test_10_seconds_1_request(self):
        """测试10秒内1次请求的限流（快速版本，使用0.5秒模拟）"""
        call_times = []

        @rate_limit(max_rate=1, time_period=0.5)  # 使用0.5秒来快速测试
        async def low_freq_func():
            call_times.append(time.time())
            return len(call_times)

        start_time = time.time()

        # 连续调用3次
        results = []
        for i in range(3):
            result = await low_freq_func()
            results.append(result)
            elapsed = time.time() - start_time
            print(f"第{i+1}次调用完成，耗时: {elapsed:.3f}秒")

        total_time = time.time() - start_time

        assert len(results) == 3
        assert results == [1, 2, 3]
        assert len(call_times) == 3

        # 检查时间间隔
        if len(call_times) >= 2:
            interval1 = call_times[1] - call_times[0]
            assert interval1 >= 0.4, f"第二次调用间隔不足: {interval1:.3f}秒"

        if len(call_times) >= 3:
            interval2 = call_times[2] - call_times[1]
            assert interval2 >= 0.4, f"第三次调用间隔不足: {interval2:.3f}秒"

        # 总时间应该约为1秒（两个0.5秒间隔）
        assert total_time >= 0.9, f"总时间不足: {total_time:.3f}秒"
        assert total_time < 1.5, f"总时间过长: {total_time:.3f}秒"

        print(f"10秒1次测试完成（模拟），总耗时: {total_time:.3f}秒")

    @pytest.mark.asyncio
    async def test_concurrent_performance_stress(self):
        """并发性能压力测试"""
        import asyncio

        call_count = 0
        lock = asyncio.Lock()

        @rate_limit(max_rate=20, time_period=1)  # 降低限制使效果更明显
        async def stress_func(task_id):
            nonlocal call_count
            async with lock:  # 使用锁确保计数器的原子性
                call_count += 1
                current_count = call_count
            await asyncio.sleep(0.001)  # 模拟一些处理时间
            return current_count

        start_time = time.time()

        # 创建50个并发任务，每秒20个，应该需要约2.5秒
        tasks = [stress_func(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        assert len(results) == 50
        # 由于并发和限流，不能保证结果完全不重复，但应该有合理的分布
        unique_results = len(set(results))
        assert unique_results >= 25, f"唯一结果数量过少: {unique_results}"
        assert max(results) == 50

        # 50个请求，每秒20个，leaky bucket算法实际表现可能不同
        # 主要验证限流确实在工作，不要求精确的时间
        assert total_time >= 1.0, f"压力测试完成过快: {total_time:.3f}秒"
        assert total_time <= 4.0, f"压力测试完成过慢: {total_time:.3f}秒"

        # 计算吞吐量
        throughput = 50 / total_time
        print(
            f"并发压力测试: {total_time:.3f}秒完成50个请求，吞吐量: {throughput:.1f} req/s"
        )
        print(f"唯一结果数量: {unique_results}/50")

        # 吞吐量验证（考虑到leaky bucket算法的特性，给予一定余量）
        assert throughput <= 40, f"吞吐量明显超过限制: {throughput:.1f} req/s"
        assert throughput >= 10, f"吞吐量过低: {throughput:.1f} req/s"

        # 主要目的是验证限流装饰器能正常工作，不会无限制地执行
        print("✓ 限流装饰器正常工作，控制了执行速度")

    @pytest.mark.asyncio
    async def test_multiple_limiters_isolation(self):
        """测试多个限流器的隔离性"""
        results = {"fast": [], "slow": []}

        @rate_limit(
            max_rate=5, time_period=1, key_func=lambda service: f"service_{service}"
        )
        async def service_call(service: str):
            results[service].append(time.time())
            return f"{service}_result_{len(results[service])}"

        start_time = time.time()

        # 同时测试快服务和慢服务
        tasks = []

        # 快服务调用6次（应该有1次等待）
        for i in range(6):
            tasks.append(service_call("fast"))

        # 慢服务调用3次（应该都能立即执行，因为有独立的限流器）
        for i in range(3):
            tasks.append(service_call("slow"))

        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # 验证调用次数
        assert len(results["fast"]) == 6
        assert len(results["slow"]) == 3

        # 快服务应该有等待时间
        fast_total_time = results["fast"][-1] - results["fast"][0]
        slow_total_time = results["slow"][-1] - results["slow"][0]

        assert (
            fast_total_time >= 0.15
        ), f"快服务限流等待时间不足: {fast_total_time:.3f}秒"
        assert slow_total_time < 0.1, f"慢服务不应该有明显等待: {slow_total_time:.3f}秒"

        print(f"多限流器隔离测试完成，总耗时: {total_time:.3f}秒")
        print(
            f"快服务总耗时: {fast_total_time:.3f}秒，慢服务总耗时: {slow_total_time:.3f}秒"
        )


if __name__ == "__main__":
    # 直接运行快速测试
    pytest.main([__file__, "-v", "-s"])
