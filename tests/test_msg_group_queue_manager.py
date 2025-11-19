#!/usr/bin/env python3
"""
MsgGroupQueueManager 重构后的完整测试套件
"""

import asyncio
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from core.queue.msg_group_queue.msg_group_queue_manager import (
    MsgGroupQueueManager,
    QueueStats,
    ManagerStats,
    ShutdownMode,
    TimeWindowStats,
)
from core.queue.msg_group_queue.msg_group_queue_manager_factory import (
    MsgGroupQueueManagerFactory,
    MsgGroupQueueConfig,
)

# ============ 基础测试函数 ============


async def test_basic_functionality():
    """基本功能测试"""
    manager = MsgGroupQueueManager("basic_test", num_queues=3, max_total_messages=10)

    # 投递测试
    success = await manager.deliver_message("test_user", {"msg": "hello"})
    assert success, "基本投递失败"

    # 消费测试
    target_queue = manager._hash_route("test_user")  # pylint: disable=protected-access
    message = await manager.get_by_queue(target_queue, wait=False)
    assert message is not None, "基本消费失败"

    # 统计测试
    stats = await manager.get_manager_stats()
    assert stats["total_delivered_messages"] == 1, "统计错误"

    await manager.shutdown(ShutdownMode.HARD)


async def test_queue_full_scenarios():
    """队列满了场景测试"""
    manager = MsgGroupQueueManager("full_test", num_queues=3, max_total_messages=10)

    # 尝试投递大量消息
    delivered_count = 0
    rejected_count = 0

    for i in range(15):
        success = await manager.deliver_message(f"user_{i}", f"msg_{i}")
        if success:
            delivered_count += 1
        else:
            rejected_count += 1

    print(f"✅ 队列满了测试通过 (投递{delivered_count}条, 拒绝{rejected_count}条)")
    await manager.shutdown(ShutdownMode.HARD)


async def test_time_window_stats():
    """时间窗口统计测试"""
    manager = MsgGroupQueueManager("time_test", num_queues=3, max_total_messages=20)

    # 投递消息
    for i in range(5):
        await manager.deliver_message(f"time_user_{i}", f"time_msg_{i}")

    # 消费部分消息
    consumed_count = 0
    for i in range(3):
        target_queue = manager._hash_route(
            f"time_user_{i}"
        )  # pylint: disable=protected-access
        message = await manager.get_by_queue(target_queue, wait=False)
        if message:
            consumed_count += 1

    # 检查时间窗口统计
    stats = await manager.get_manager_stats()
    assert stats["delivered_1min"] >= 5, "1分钟投递统计错误"
    assert stats["consumed_1min"] >= consumed_count, "1分钟消费统计错误"

    print(f"✅ 时间窗口统计测试通过 (投递5条, 消费{consumed_count}条)")
    await manager.shutdown(ShutdownMode.HARD)


async def test_basic_concurrent_operations():
    """基本并发操作测试"""
    manager = MsgGroupQueueManager(
        "concurrent_test", num_queues=5, max_total_messages=50
    )

    async def producer(producer_id: int):
        for i in range(10):
            await manager.deliver_message(
                f"producer_{producer_id}_msg_{i}", f"data_{i}"
            )

    async def consumer():
        consumed = 0
        for queue_id in range(manager.num_queues):
            for _ in range(5):  # 每个队列尝试消费5次
                message = await manager.get_by_queue(queue_id, wait=False)
                if message:
                    consumed += 1
        return consumed

    # 启动3个生产者
    producers = [producer(i) for i in range(3)]
    await asyncio.gather(*producers)

    # 启动消费者
    consumed_count = await consumer()

    stats = await manager.get_manager_stats()
    assert (
        stats["total_delivered_messages"] == 30
    ), f"并发投递统计错误: {stats['total_delivered_messages']}"

    print(f"✅ 并发操作测试通过 (生产30条, 消费{consumed_count}条)")
    await manager.shutdown(ShutdownMode.HARD)


async def test_basic_shutdown_modes():
    """基本Shutdown模式测试"""
    manager = MsgGroupQueueManager(
        "basic_shutdown_test", num_queues=3, max_total_messages=10
    )

    # 确保shutdown状态是干净的
    manager._shutdown_state.reset()  # pylint: disable=protected-access

    # 添加一些消息
    for i in range(5):
        await manager.deliver_message(f"user_{i}", f"msg_{i}")

    # 测试软关闭（需要轮询直到成功）
    start_time = time.time()

    # 第一次调用软关闭，应该立即返回False
    result = await manager.shutdown(ShutdownMode.SOFT, max_delay_seconds=1.0)
    first_call_time = time.time()

    assert result is False, "第一次软关闭应该立即返回False（有消息未处理）"
    assert first_call_time - start_time < 0.1, "第一次软关闭应该立即返回"
    print("  第一次软关闭正确返回False（立即返回）")

    # 轮询直到软关闭成功或超时
    max_poll_time = start_time + 2.0  # 最多轮询2秒
    final_result = False

    while time.time() < max_poll_time:
        await asyncio.sleep(0.1)  # 等待100ms再次检查
        result = await manager.shutdown(ShutdownMode.SOFT)  # 不需要再次设置delay
        if result is True:
            final_result = True
            break

    end_time = time.time()
    total_elapsed = end_time - start_time

    # 验证最终结果和时间
    assert final_result is True, "软关闭最终应该成功"
    assert (
        0.8 <= total_elapsed <= 1.5
    ), f"软关闭总时间异常: {total_elapsed:.2f}秒，期望约1.0秒"

    print(f"  软关闭轮询成功，总耗时: {total_elapsed:.2f}秒")


async def test_edge_cases():
    """边界情况测试"""
    manager = MsgGroupQueueManager("edge_test", num_queues=3, max_total_messages=10)

    # 测试1: 空字符串group_key
    success = await manager.deliver_message("", "empty_key_msg")
    assert success, "空字符串key投递失败"

    empty_queue = manager._hash_route("")  # pylint: disable=protected-access
    result = await manager.get_by_queue(empty_queue, wait=False)
    assert result is not None, "空字符串key消费失败"
    key, data = result
    assert key == "", f"空字符串key不匹配: {key}"
    assert data == "empty_key_msg", f"空字符串key数据不匹配: {data}"

    # 测试2: None消息数据
    success = await manager.deliver_message("none_test", None)
    assert success, "None消息投递失败"

    none_queue = manager._hash_route("none_test")  # pylint: disable=protected-access
    result = await manager.get_by_queue(none_queue, wait=False)
    assert result is not None, "None消息消费失败"
    key, data = result
    assert key == "none_test", f"None消息key不匹配: {key}"
    assert data is None, f"None消息数据不匹配: {data}"

    # 测试3: 复杂数据
    complex_data = {"nested": {"list": [1, 2, 3]}, "unicode": "测试🎉"}
    success = await manager.deliver_message("complex_test", complex_data)
    assert success, "复杂数据投递失败"

    complex_queue = manager._hash_route(
        "complex_test"
    )  # pylint: disable=protected-access
    result = await manager.get_by_queue(complex_queue, wait=False)
    assert result is not None, "复杂数据消费失败"
    key, data = result
    assert key == "complex_test", f"复杂数据key不匹配: {key}"
    assert data == complex_data, f"复杂数据不匹配: 期望={complex_data}, 实际={data}"

    await manager.shutdown(ShutdownMode.HARD)


async def test_factory_pattern():
    """工厂模式测试"""
    factory = MsgGroupQueueManagerFactory()

    # 默认管理器
    manager1 = await factory.get_default_manager(auto_start=False)
    manager2 = await factory.get_default_manager(auto_start=False)
    assert manager1 is manager2, "默认管理器应该是单例"

    # 自定义配置
    config = MsgGroupQueueConfig(name="custom", num_queues=5, max_total_messages=25)
    manager3 = await factory.get_manager(config, auto_start=False)
    assert manager3.name == "custom", "自定义配置错误"

    # 命名管理器
    manager4 = await factory.get_named_manager("test_named", auto_start=False)
    assert manager4.name == "test_named", "命名管理器错误"

    await factory.stop_all_managers()


async def test_timeout_mechanism():
    """超时机制测试"""
    manager = MsgGroupQueueManager("timeout_test", num_queues=3, max_total_messages=10)

    try:
        # 测试超时获取（空队列）
        start_time = time.time()
        try:
            result = await manager.get_by_queue(0, wait=True, timeout=0.5)
            end_time = time.time()

            # 应该超时返回None或抛出TimeoutError
            if result is not None:
                raise AssertionError("超时应该返回None或抛出异常")
            assert (
                0.4 <= end_time - start_time <= 0.6
            ), f"超时时间不准确: {end_time - start_time}"

        except asyncio.TimeoutError:
            end_time = time.time()
            assert (
                0.4 <= end_time - start_time <= 0.6
            ), f"超时时间不准确: {end_time - start_time}"

        # 测试有消息时的获取
        await manager.deliver_message("timeout_user", "timeout_msg")
        target_queue = manager._hash_route(
            "timeout_user"
        )  # pylint: disable=protected-access

        start_time = time.time()
        result = await manager.get_by_queue(target_queue, wait=True, timeout=1.0)
        end_time = time.time()

        # 应该立即返回消息
        assert result is not None, "有消息时应该立即返回"
        assert (
            end_time - start_time < 0.1
        ), f"有消息时不应该等待: {end_time - start_time}"

        key, data = result
        assert key == "timeout_user", "超时测试消息key错误"
        assert data == "timeout_msg", "超时测试消息数据错误"

    finally:
        await manager.shutdown(ShutdownMode.HARD)


async def test_routing_uniformity():
    """路由均匀性测试 - 使用随机UUID验证hash分布"""
    manager = MsgGroupQueueManager(
        "routing_test", num_queues=10, max_total_messages=2000
    )

    try:
        # 生成大量随机UUID作为group_key
        test_count = 1000
        uuid_keys = [str(uuid.uuid4()) for _ in range(test_count)]

        # 统计每个队列收到的消息数
        queue_counts = defaultdict(int)

        # 投递所有消息并统计路由分布
        for i, group_key in enumerate(uuid_keys):
            success = await manager.deliver_message(group_key, f"message_{i}")
            assert success, f"UUID消息投递失败: {group_key}"

            # 计算这个key路由到哪个队列
            target_queue = manager._hash_route(
                group_key
            )  # pylint: disable=protected-access
            queue_counts[target_queue] += 1

        # 分析分布均匀性
        print(f"📊 路由分布统计 ({test_count}个UUID):")
        expected_per_queue = test_count / manager.num_queues

        total_deviation = 0
        max_count = 0
        min_count = test_count

        for queue_id in range(manager.num_queues):
            count = queue_counts[queue_id]
            percentage = (count / test_count) * 100
            deviation = abs(count - expected_per_queue)
            deviation_percent = (deviation / expected_per_queue) * 100

            print(
                f"   队列[{queue_id}]: {count:3d}条 ({percentage:5.1f}%) - 偏差: {deviation_percent:5.1f}%"
            )

            total_deviation += deviation
            max_count = max(max_count, count)
            min_count = min(min_count, count)

        # 计算分布质量指标
        avg_count = test_count / manager.num_queues
        variance = (
            sum((queue_counts[i] - avg_count) ** 2 for i in range(manager.num_queues))
            / manager.num_queues
        )
        std_dev = variance**0.5
        coefficient_of_variation = (std_dev / avg_count) * 100

        print(f"\n📈 分布质量分析:")
        print(f"   期望每队列: {expected_per_queue:.1f}条")
        print(f"   实际范围: {min_count}-{max_count}条")
        print(f"   标准差: {std_dev:.2f}")
        print(f"   变异系数: {coefficient_of_variation:.1f}%")

        # 验证分布质量
        # 1. 变异系数应该小于15%（较好的均匀性）
        assert (
            coefficient_of_variation < 15.0
        ), f"分布不够均匀，变异系数: {coefficient_of_variation:.1f}%"

        # 2. 最大值和最小值的差异不应该太大
        max_min_ratio = max_count / min_count if min_count > 0 else float('inf')
        assert (
            max_min_ratio < 2.0
        ), f"队列负载差异过大，最大/最小比率: {max_min_ratio:.2f}"

        # 3. 每个队列都应该有消息
        assert min_count > 0, "存在空队列，分布有问题"

        # 验证路由一致性 - 同一个UUID总是路由到同一个队列
        print(f"\n🔍 验证路由一致性...")
        consistency_test_keys = uuid_keys[:50]  # 取前50个UUID测试一致性

        for test_key in consistency_test_keys:
            # 多次计算同一个key的路由，应该总是相同
            routes = [
                manager._hash_route(test_key) for _ in range(10)
            ]  # pylint: disable=protected-access
            assert (
                len(set(routes)) == 1
            ), f"路由不一致: key={test_key}, routes={set(routes)}"

        print(f"   ✅ {len(consistency_test_keys)}个UUID路由一致性验证通过")

        # 验证统计信息
        stats = await manager.get_manager_stats()
        assert stats["total_delivered_messages"] == test_count, "投递统计错误"

        print(f"\n✅ 路由均匀性测试通过:")
        print(f"   - 变异系数: {coefficient_of_variation:.1f}% (< 15%)")
        print(f"   - 负载比率: {max_min_ratio:.2f} (< 2.0)")
        print(f"   - 路由一致性: 100%")

    finally:
        await manager.shutdown(ShutdownMode.HARD)


# ============ 扩展测试函数 ============


async def test_concurrent_operations():
    """并发操作详细测试"""
    manager = MsgGroupQueueManager(
        "concurrent_test", num_queues=5, max_total_messages=100
    )

    # 定义生产者和消费者任务
    async def producer(producer_id: int, message_count: int):
        for i in range(message_count):
            group_key = f"producer_{producer_id}_user_{i % 10}"  # 10个不同的用户
            message_data = {"producer": producer_id, "seq": i, "data": f"message_{i}"}
            success = await manager.deliver_message(group_key, message_data)
            if not success:
                print(f"生产者{producer_id}第{i}条消息投递失败")
            await asyncio.sleep(0.01)  # 小延迟模拟真实场景

    async def consumer(consumer_id: int, target_queues: List[int]):
        consumed = 0
        for queue_id in target_queues:
            while True:
                try:
                    message = await manager.get_by_queue(
                        queue_id, wait=True, timeout=0.1
                    )
                    if message is None:
                        break
                    consumed += 1
                    await asyncio.sleep(0.005)  # 模拟处理时间
                except asyncio.TimeoutError:
                    break
        return consumed

    # 启动3个生产者，每个生产5条消息
    producers = [producer(i, 5) for i in range(3)]
    await asyncio.gather(*producers)

    # 启动2个消费者，分别处理不同的队列
    consumer_tasks = [
        consumer(0, [0, 1, 2]),  # 消费者0处理队列0,1,2
        consumer(1, [3, 4]),  # 消费者1处理队列3,4
    ]
    consumed_counts = await asyncio.gather(*consumer_tasks)

    # 验证结果
    total_consumed = sum(consumed_counts)
    stats = await manager.get_manager_stats()

    assert (
        stats["total_delivered_messages"] == 15
    ), f"投递数量错误: {stats['total_delivered_messages']}"
    assert total_consumed <= 15, f"消费数量异常: {total_consumed}"

    await manager.shutdown(ShutdownMode.HARD)


async def test_real_world_scenario():
    """真实场景测试（模拟Kafka消息处理）"""
    print("🚀 模拟消息爆发...")
    manager = MsgGroupQueueManager(
        "kafka_simulator", num_queues=10, max_total_messages=200
    )

    # 模拟消息爆发
    delivered = 0
    for i in range(100):
        user_id = f"user_{i % 20}"  # 20个不同用户
        message = {
            "user_id": user_id,
            "timestamp": time.time(),
            "event_type": "click" if i % 3 == 0 else "view",
            "data": {"page": f"page_{i % 5}", "session": f"session_{i % 10}"},
        }
        success = await manager.deliver_message(user_id, message)
        if success:
            delivered += 1

    print(f"📊 成功投递 {delivered} 条消息")

    # 模拟消费者处理
    print("🔄 模拟消费者处理...")
    total_consumed = 0

    # 启动多个消费者并发处理
    async def consumer_worker(queue_ids: List[int]):
        consumed = 0
        for queue_id in queue_ids:
            while True:
                try:
                    message = await manager.get_by_queue(
                        queue_id, wait=True, timeout=0.05
                    )
                    if message is None:
                        break
                    consumed += 1
                    await asyncio.sleep(0.001)  # 快速处理
                except asyncio.TimeoutError:
                    break
        return consumed

    # 启动5个消费者工作线程
    consumer_tasks = [
        consumer_worker([0, 1]),
        consumer_worker([2, 3]),
        consumer_worker([4, 5]),
        consumer_worker([6, 7]),
        consumer_worker([8, 9]),
    ]

    consumed_counts = await asyncio.gather(*consumer_tasks)
    total_consumed = sum(consumed_counts)

    print(f"📊 总共消费 {total_consumed} 条消息")

    # 检查队列负载分布
    queue_info = await manager.get_queue_info()
    print("📈 队列负载分布:")
    for info in queue_info:
        if info["current_size"] > 0:
            print(f"   队列[{info['queue_id']}]: {info['current_size']} 条消息")

    # 测试高负载下的投递拒绝
    print("⚠️ 测试高负载下的投递拒绝...")
    rejected = 0
    for i in range(100, 200):
        success = await manager.deliver_message(f"flood_user_{i}", f"flood_msg_{i}")
        if not success:
            rejected += 1

    # 清理
    print("🧹 清理剩余消息...")
    cleaned = 0
    for queue_id in range(manager.num_queues):
        while True:
            message = await manager.get_by_queue(queue_id, wait=False)
            if message is None:
                break
            cleaned += 1

    print(f"🧹 清理了 {cleaned} 条剩余消息")
    print("✅ 真实场景测试完成!")

    await manager.shutdown(ShutdownMode.HARD)


async def test_queue_overflow_and_recovery():
    """队列溢出恢复测试"""
    manager = MsgGroupQueueManager("overflow_test", num_queues=3, max_total_messages=15)

    # 填满队列
    print("📦 填满队列...")
    delivered = 0
    rejected = 0

    for i in range(30):
        success = await manager.deliver_message(f"user_{i % 5}", f"msg_{i}")
        if success:
            delivered += 1
        else:
            rejected += 1

    print(f"📊 投递统计: 成功={delivered}, 拒绝={rejected}")

    # 部分消费以恢复投递能力
    print("🔄 部分消费恢复投递能力...")
    consumed = 0
    for queue_id in range(manager.num_queues):
        # 每个队列消费几条消息
        for _ in range(2):
            message = await manager.get_by_queue(queue_id, wait=False)
            if message:
                consumed += 1

    print(f"📤 消费了 {consumed} 条消息用于恢复")

    # 尝试再次投递
    recovery_delivered = 0
    for i in range(5):
        success = await manager.deliver_message(
            f"recovery_user_{i}", f"recovery_msg_{i}"
        )
        if success:
            recovery_delivered += 1

    print(f"✅ 恢复投递了 {recovery_delivered} 条消息")

    await manager.shutdown(ShutdownMode.HARD)


async def test_shutdown_modes_integration():
    """测试shutdown模式的集成场景"""
    manager = MsgGroupQueueManager("shutdown_test", num_queues=3, max_total_messages=20)

    try:
        # 添加一些消息
        for i in range(5):
            await manager.deliver_message(f"user_{i}", f"msg_{i}")

        # 启动一个消费者任务（模拟等待中的消费）
        async def consumer_task():
            consumed = 0
            for _ in range(10):  # 尝试消费多次
                try:
                    message = await manager.get_by_queue(1, wait=True, timeout=0.5)
                    if message:
                        consumed += 1
                        await asyncio.sleep(0.1)  # 模拟处理时间
                except asyncio.TimeoutError:
                    break
            return consumed

        # 启动shutdown任务
        async def shutdown_task():
            await asyncio.sleep(1.0)  # 等待消费者开始
            result = await manager.shutdown(ShutdownMode.SOFT, max_delay_seconds=3.0)
            return result

        # 等待任务完成
        results = await asyncio.gather(
            shutdown_task(), consumer_task(), return_exceptions=True
        )

        shutdown_result, consumed_count = results[0], results[1]

        # 验证结果
        if isinstance(shutdown_result, Exception):
            raise shutdown_result
        if isinstance(consumed_count, Exception):
            raise consumed_count

        # shutdown_result可能是bool或dict
        assert isinstance(
            shutdown_result, (bool, dict)
        ), f"Shutdown结果类型错误: {type(shutdown_result)}"
        assert isinstance(
            consumed_count, int
        ), f"消费计数类型错误: {type(consumed_count)}"
        assert consumed_count >= 0, f"消费计数异常: {consumed_count}"

    except Exception:
        # 确保在异常情况下也能清理资源
        try:
            await manager.shutdown(ShutdownMode.HARD)
        except Exception:
            pass  # 忽略清理时的错误
        raise


# ============ 主测试运行器 ============

if __name__ == "__main__":

    async def run_all_tests():
        print("🚀 开始运行 MsgGroupQueueManager 重构测试套件...")
        print("=" * 60)

        test_results = []

        # 基础测试套件
        basic_tests = [
            ("基本功能测试", test_basic_functionality),
            ("队列满了测试", test_queue_full_scenarios),
            ("时间窗口统计测试", test_time_window_stats),
            ("基本并发操作测试", test_basic_concurrent_operations),
            ("基本Shutdown模式测试", test_basic_shutdown_modes),
            ("边界情况测试", test_edge_cases),
            ("工厂模式测试", test_factory_pattern),
            ("超时机制测试", test_timeout_mechanism),
            ("路由均匀性测试", test_routing_uniformity),
        ]

        # 扩展测试套件
        extended_tests = [
            ("并发操作详细测试", test_concurrent_operations),
            ("真实场景测试", test_real_world_scenario),
            ("队列溢出恢复测试", test_queue_overflow_and_recovery),
            ("Shutdown模式集成测试", test_shutdown_modes_integration),
        ]

        # 运行所有测试
        all_tests = basic_tests + extended_tests

        for i, (test_name, test_func) in enumerate(all_tests, 1):
            print(f"\n{i}️⃣ {test_name}...")
            try:
                await test_func()
                test_results.append((test_name, "✅ 通过"))
                print(f"✅ {test_name}通过")
            except Exception as e:
                test_results.append((test_name, f"❌ 失败: {e}"))
                print(f"❌ {test_name}失败: {e}")

        # 测试结果汇总
        print("\n" + "=" * 60)
        print("📊 测试结果汇总:")
        print("=" * 60)

        passed_count = 0
        total_count = len(test_results)

        for test_name, result in test_results:
            print(f"{result:<20} {test_name}")
            if "✅" in result:
                passed_count += 1

        print("=" * 60)
        print(f"📈 总计: {passed_count}/{total_count} 测试通过")

        if passed_count == total_count:
            print("🎉 所有测试都通过了！MsgGroupQueueManager 工作正常！")
        else:
            print(
                f"⚠️  有 {total_count - passed_count} 个测试失败，请检查上面的错误信息"
            )

        return passed_count == total_count

    # 运行所有测试
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
