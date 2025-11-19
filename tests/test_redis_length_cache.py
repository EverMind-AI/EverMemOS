"""
Redis 长度限制缓存管理器测试脚本

使用方法:
    python src/bootstrap.py tests/test_redis_length_cache.py

测试覆盖：
1. 基本操作：追加、获取、清理等
2. 长度限制清理机制
3. 时间戳兼容性（整数和datetime对象）
4. 按时间戳范围获取数据功能
5. 压力测试（大量数据处理）
6. 过期机制测试
7. 向后兼容层测试
"""

import asyncio
import time
from datetime import datetime, timezone
from core.di.utils import get_bean
from core.observation.logger import get_logger

logger = get_logger(__name__)


class NonSerializableTestClass:
    """不能JSON序列化的测试类，用于测试Pickle降级"""

    def __init__(self, name, value, multiplier=2):
        self.name = name
        self.value = value
        self.multiplier = multiplier
        self.created_at = time.time()
        # 添加一些复杂的属性让JSON序列化失败
        self.complex_data = {
            "set_data": {1, 2, 3, 4, 5},  # set不能JSON序列化
            "tuple_data": (1, 2, 3),  # tuple会变成list，但我们可以检测
            "bytes_data": b"hello world",  # bytes不能JSON序列化
        }

    def get_doubled_value(self):
        return self.value * self.multiplier

    def process_data(self, input_value):
        """处理数据的方法"""
        return f"{self.name}_processed_{input_value}_{self.multiplier}"

    def __eq__(self, other):
        return (
            isinstance(other, NonSerializableTestClass)
            and self.name == other.name
            and self.value == other.value
            and self.multiplier == other.multiplier
        )

    def __repr__(self):
        return f"NonSerializableTestClass(name='{self.name}', value={self.value}, multiplier={self.multiplier})"


async def test_basic_operations():
    """测试基本操作：追加、获取、清理等"""
    logger.info("开始测试基本操作...")

    # 从DI容器获取缓存管理器工厂
    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=5, expire_minutes=5
    )  # 最大5条，5分钟过期

    test_key = "test_length_cache"

    # 1. 清空测试队列
    logger.info("清空测试队列...")
    await cache.clear_queue(test_key)

    # 2. 测试追加数据
    logger.info("测试追加数据...")
    test_data = [
        "test_string",
        {"name": "test_dict", "value": 123},
        ["test_list", 1, 2, 3],
        42,
        {"complex": {"nested": "data"}},
    ]

    for i, data in enumerate(test_data):
        success = await cache.append(test_key, data)
        assert success, f"追加数据失败: {data}"
        logger.info("追加第 %d 条数据成功", i + 1)

    # 3. 验证队列大小
    size = await cache.get_queue_size(test_key)
    assert size == len(test_data), f"队列大小不匹配: 期望 {len(test_data)}, 实际 {size}"
    logger.info("队列大小验证通过: %d", size)

    # 4. 获取队列统计信息
    stats = await cache.get_queue_stats(test_key)
    assert isinstance(stats, dict), "统计信息格式错误"
    assert stats["total_count"] == len(test_data), "统计信息数量不匹配"
    assert stats["max_length"] == 5, "最大长度配置不匹配"
    assert stats["is_full"], "队列应该是满的（5/5）"
    logger.info("队列统计信息: %s", stats)

    logger.info("✅ 基本操作测试通过")


async def test_length_cleanup():
    """测试长度限制清理机制"""
    logger.info("开始测试长度限制清理...")

    factory = get_bean("redis_length_cache_factory")
    # 设置最大长度为3，清理概率为0确保不会自动清理，方便我们手动验证
    cache = await factory.create_cache_manager(
        max_length=3, expire_minutes=10, cleanup_probability=0.0
    )

    test_key = "test_length_cleanup"

    # 1. 清空测试队列
    await cache.clear_queue(test_key)

    # 2. 添加数据，使用明确的时间戳确保顺序
    logger.info("添加数据，使用递增时间戳...")
    base_timestamp = int(time.time() * 1000)
    data_list = []

    for i in range(5):  # 添加5条数据，超过最大长度3
        timestamp = base_timestamp + i * 1000  # 每条数据间隔1秒
        data = {"index": i, "content": f"data_{i}", "timestamp": timestamp}
        data_list.append({"data": data, "timestamp": timestamp})

        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"追加第 {i+1} 条数据失败"
        logger.info("添加数据 %d: timestamp=%d, content=data_%d", i, timestamp, i)

    # 3. 验证所有数据都已添加
    current_size = await cache.get_queue_size(test_key)
    logger.info("添加完成后队列大小: %d", current_size)
    assert current_size == 5, f"期望队列大小为5，实际为{current_size}"

    # 4. 获取当前队列中的所有数据，验证顺序
    logger.info("获取队列中的数据，验证是否按时间戳排序...")
    # 注意：由于没有get_all方法，我们通过统计信息来验证时间范围
    stats = await cache.get_queue_stats(test_key)
    logger.info(
        "队列统计信息: oldest_timestamp=%s, newest_timestamp=%s",
        stats.get("oldest_timestamp"),
        stats.get("newest_timestamp"),
    )

    # 验证最早和最新的时间戳
    expected_oldest = base_timestamp
    expected_newest = base_timestamp + 4 * 1000
    assert (
        stats["oldest_timestamp"] == expected_oldest
    ), f"最早时间戳不匹配: 期望{expected_oldest}, 实际{stats['oldest_timestamp']}"
    assert (
        stats["newest_timestamp"] == expected_newest
    ), f"最新时间戳不匹配: 期望{expected_newest}, 实际{stats['newest_timestamp']}"

    # 5. 手动触发清理，应该删除最早的2条数据（保留3条）
    logger.info("手动触发清理，应该删除最早的2条数据...")
    cleaned_count = await cache.cleanup_excess(test_key)
    logger.info("手动清理完成，清理数量: %d", cleaned_count)
    assert cleaned_count == 2, f"期望清理2条数据，实际清理了{cleaned_count}条"

    # 6. 验证清理后的数据
    final_size = await cache.get_queue_size(test_key)
    assert final_size == 3, f"清理后队列大小应为3，实际为{final_size}"

    # 7. 验证剩余的是最新的3条数据（index 2, 3, 4）
    stats_after = await cache.get_queue_stats(test_key)
    expected_oldest_after = base_timestamp + 2 * 1000  # data_2的时间戳
    expected_newest_after = base_timestamp + 4 * 1000  # data_4的时间戳

    logger.info(
        "清理后队列统计信息: oldest_timestamp=%s, newest_timestamp=%s",
        stats_after.get("oldest_timestamp"),
        stats_after.get("newest_timestamp"),
    )

    assert (
        stats_after["oldest_timestamp"] == expected_oldest_after
    ), f"清理后最早时间戳应为{expected_oldest_after}，实际为{stats_after['oldest_timestamp']}"
    assert (
        stats_after["newest_timestamp"] == expected_newest_after
    ), f"清理后最新时间戳应为{expected_newest_after}，实际为{stats_after['newest_timestamp']}"

    logger.info(
        "✅ 验证通过：清理删除的是最早的数据（data_0和data_1），保留了最新的数据（data_2、data_3、data_4）"
    )

    # 8. 再次清理，应该没有数据被删除（因为已经是最大长度）
    additional_cleaned = await cache.cleanup_excess(test_key)
    assert (
        additional_cleaned == 0
    ), f"再次清理应该删除0条数据，实际删除了{additional_cleaned}条"

    logger.info("✅ 长度限制清理测试通过")


async def test_timestamp_compatibility():
    """测试时间戳兼容性：支持整数和datetime对象"""
    logger.info("开始测试时间戳兼容性...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(max_length=10, expire_minutes=5)

    test_key = "test_timestamp_compat"

    # 1. 清空测试队列
    await cache.clear_queue(test_key)

    # 2. 测试不同类型的时间戳
    logger.info("测试不同类型的时间戳...")

    # 使用当前时间（默认）
    success = await cache.append(test_key, "data_current_time")
    assert success, "使用当前时间追加失败"

    # 使用整数时间戳（毫秒）
    timestamp_ms = int(time.time() * 1000) + 1000  # 1秒后
    success = await cache.append(test_key, "data_int_timestamp", timestamp=timestamp_ms)
    assert success, "使用整数时间戳追加失败"

    # 使用datetime对象（无时区）
    dt_naive = datetime.now()
    success = await cache.append(test_key, "data_datetime_naive", timestamp=dt_naive)
    assert success, "使用datetime对象追加失败"

    # 使用datetime对象（带时区）
    dt_with_tz = datetime.now(timezone.utc)
    success = await cache.append(test_key, "data_datetime_tz", timestamp=dt_with_tz)
    assert success, "使用带时区datetime对象追加失败"

    # 3. 验证数据都被正确存储
    size = await cache.get_queue_size(test_key)
    assert size == 4, f"时间戳兼容性测试数据数量不匹配: 期望 4, 实际 {size}"

    # 4. 获取统计信息验证时间戳
    stats = await cache.get_queue_stats(test_key)
    assert stats["oldest_timestamp"] is not None, "最旧时间戳为空"
    assert stats["newest_timestamp"] is not None, "最新时间戳为空"
    assert stats["oldest_datetime"] is not None, "最旧时间字符串为空"
    assert stats["newest_datetime"] is not None, "最新时间字符串为空"

    logger.info(
        "时间戳范围: %s 到 %s", stats["oldest_datetime"], stats["newest_datetime"]
    )
    logger.info("✅ 时间戳兼容性测试通过")


async def test_timestamp_range_query():
    """测试按时间戳范围获取数据功能"""
    logger.info("开始测试按时间戳范围获取数据...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=20, expire_minutes=10, cleanup_probability=0.0
    )

    test_key = "test_timestamp_range"

    # 1. 清空测试队列
    await cache.clear_queue(test_key)

    # 2. 添加测试数据，使用明确的时间戳
    logger.info("添加测试数据，使用明确的时间戳...")
    base_timestamp = int(time.time() * 1000)
    test_data = []

    for i in range(10):
        timestamp = base_timestamp + i * 10000  # 每条数据间隔10秒
        data = {"index": i, "content": f"data_{i}", "created_at": timestamp}
        test_data.append({"data": data, "timestamp": timestamp})

        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"添加第 {i+1} 条数据失败"

    logger.info(
        "添加了 %d 条数据，时间范围: %d 到 %d",
        len(test_data),
        base_timestamp,
        base_timestamp + 9 * 10000,
    )

    # 3. 测试获取所有数据（不限制时间范围）
    all_data = await cache.get_by_timestamp_range(test_key)
    assert len(all_data) == 10, f"获取所有数据失败，期望10条，实际{len(all_data)}条"
    logger.info("获取所有数据成功: %d 条", len(all_data))

    # 4. 测试按开始时间过滤（获取第5条数据之后的）
    start_time = base_timestamp + 4 * 10000  # data_4的时间戳
    filtered_data = await cache.get_by_timestamp_range(
        test_key, start_timestamp=start_time
    )
    assert (
        len(filtered_data) == 6
    ), f"按开始时间过滤失败，期望6条，实际{len(filtered_data)}条"  # data_4到data_9

    # 验证数据顺序（应该是最新的在前）
    assert filtered_data[0]["data"]["index"] == 9, "数据排序错误，应该最新的在前"
    assert filtered_data[-1]["data"]["index"] == 4, "数据排序错误，最旧的应该在后"
    logger.info("按开始时间过滤测试通过")

    # 5. 测试按结束时间过滤（获取第5条数据之前的）
    end_time = base_timestamp + 4 * 10000  # data_4的时间戳
    filtered_data = await cache.get_by_timestamp_range(test_key, end_timestamp=end_time)
    assert (
        len(filtered_data) == 5
    ), f"按结束时间过滤失败，期望5条，实际{len(filtered_data)}条"  # data_0到data_4

    # 验证数据内容
    assert filtered_data[0]["data"]["index"] == 4, "按结束时间过滤结果错误"
    assert filtered_data[-1]["data"]["index"] == 0, "按结束时间过滤结果错误"
    logger.info("按结束时间过滤测试通过")

    # 6. 测试时间范围过滤（获取中间的数据）
    start_time = base_timestamp + 2 * 10000  # data_2的时间戳
    end_time = base_timestamp + 6 * 10000  # data_6的时间戳
    range_data = await cache.get_by_timestamp_range(
        test_key, start_timestamp=start_time, end_timestamp=end_time
    )
    assert (
        len(range_data) == 5
    ), f"时间范围过滤失败，期望5条，实际{len(range_data)}条"  # data_2到data_6

    # 验证数据范围
    indexes = [item["data"]["index"] for item in range_data]
    indexes.sort()  # 排序以便验证
    assert indexes == [
        2,
        3,
        4,
        5,
        6,
    ], f"时间范围过滤结果错误，期望[2,3,4,5,6]，实际{indexes}"
    logger.info("时间范围过滤测试通过")

    # 7. 测试限制数量
    limited_data = await cache.get_by_timestamp_range(test_key, limit=3)
    assert len(limited_data) == 3, f"限制数量失败，期望3条，实际{len(limited_data)}条"
    logger.info("限制数量测试通过")

    # 8. 测试使用datetime对象作为时间戳
    dt_start = datetime.fromtimestamp((base_timestamp + 3 * 10000) / 1000)
    dt_end = datetime.fromtimestamp((base_timestamp + 7 * 10000) / 1000)
    dt_data = await cache.get_by_timestamp_range(
        test_key, start_timestamp=dt_start, end_timestamp=dt_end
    )
    assert (
        len(dt_data) == 5
    ), f"使用datetime对象过滤失败，期望5条，实际{len(dt_data)}条"  # data_3到data_7
    logger.info("使用datetime对象过滤测试通过")

    # 9. 测试空结果
    future_time = base_timestamp + 20 * 10000  # 未来的时间
    empty_data = await cache.get_by_timestamp_range(
        test_key, start_timestamp=future_time
    )
    assert (
        len(empty_data) == 0
    ), f"空结果测试失败，应该返回0条数据，实际{len(empty_data)}条"
    logger.info("空结果测试通过")

    # 10. 清理测试数据
    success = await cache.clear_queue(test_key)
    assert success, "清理测试数据失败"

    logger.info("✅ 按时间戳范围获取数据测试通过")


async def test_json_pickle_mixed_data():
    """测试JSON和Pickle混合数据处理"""
    logger.info("开始测试JSON和Pickle混合数据处理...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=20, expire_minutes=10, cleanup_probability=0.0
    )

    test_key = "test_mixed_serialization"

    # 1. 清空测试队列
    await cache.clear_queue(test_key)

    # 2. 准备混合测试数据
    base_timestamp = int(time.time() * 1000)
    test_data = []

    # JSON可序列化的数据
    json_data = [
        {"type": "json", "name": "dict_data", "value": 123, "items": [1, 2, 3]},
        ["json_list", "with", "strings", 456],
        "simple_json_string",
        42,
        {"nested": {"deep": {"data": "json_nested"}}},
    ]

    # Pickle序列化的数据（不能JSON序列化）
    pickle_data = [
        NonSerializableTestClass("test1", 100),
        NonSerializableTestClass("test2", 200, 3),
        {
            "complex_set": {1, 2, 3, 4, 5},
            "bytes_data": b"binary_data",
            "name": "complex_dict",
        },  # 包含set和bytes的字典
    ]

    # 3. 添加JSON数据
    logger.info("添加JSON可序列化数据...")
    for i, data in enumerate(json_data):
        timestamp = base_timestamp + i * 1000
        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"添加JSON数据失败: {data}"
        test_data.append({"type": "json", "data": data, "timestamp": timestamp})
        logger.debug("添加JSON数据成功: %s", str(data)[:50])

    # 4. 添加Pickle数据
    logger.info("添加Pickle序列化数据...")
    for i, data in enumerate(pickle_data):
        timestamp = base_timestamp + (len(json_data) + i) * 1000
        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"添加Pickle数据失败: {data}"
        test_data.append({"type": "pickle", "data": data, "timestamp": timestamp})
        logger.debug("添加Pickle数据成功: %s", str(data)[:50])

    # 5. 验证总数据量
    total_count = len(json_data) + len(pickle_data)
    size = await cache.get_queue_size(test_key)
    assert size == total_count, f"数据总量不匹配: 期望{total_count}, 实际{size}"
    logger.info(
        "数据添加完成，总计: %d 条 (JSON: %d, Pickle: %d)",
        total_count,
        len(json_data),
        len(pickle_data),
    )

    # 6. 获取所有数据并验证
    all_data = await cache.get_by_timestamp_range(test_key)
    assert (
        len(all_data) == total_count
    ), f"获取数据数量不匹配: 期望{total_count}, 实际{len(all_data)}"

    # 7. 验证每种数据类型的正确性
    json_count = 0
    pickle_count = 0

    for item in all_data:
        retrieved_data = item["data"]

        # 检查JSON数据
        if any(
            str(retrieved_data) == str(original["data"])
            for original in test_data
            if original["type"] == "json"
        ):
            json_count += 1
            logger.debug("验证JSON数据成功: %s", str(retrieved_data)[:50])

        # 检查Pickle数据
        elif isinstance(retrieved_data, NonSerializableTestClass):
            pickle_count += 1
            # 验证Pickle对象的功能
            doubled = retrieved_data.get_doubled_value()
            expected = retrieved_data.value * retrieved_data.multiplier
            assert doubled == expected, f"Pickle对象功能异常: {doubled} != {expected}"

            # 验证复杂数据
            assert "set_data" in retrieved_data.complex_data, "Pickle对象缺少set数据"
            assert (
                "bytes_data" in retrieved_data.complex_data
            ), "Pickle对象缺少bytes数据"

            logger.debug(
                "验证Pickle对象成功: %s, 功能测试: %d * %d = %d",
                retrieved_data,
                retrieved_data.value,
                retrieved_data.multiplier,
                doubled,
            )

        elif isinstance(retrieved_data, dict) and "complex_set" in retrieved_data:
            pickle_count += 1
            # 验证包含set和bytes的字典
            assert isinstance(retrieved_data["complex_set"], set), "set数据类型错误"
            assert isinstance(retrieved_data["bytes_data"], bytes), "bytes数据类型错误"
            logger.debug("验证包含复杂数据的字典成功: %s", retrieved_data["name"])

        else:
            logger.warning("未识别的数据类型: %s", type(retrieved_data))

    # 8. 验证数据类型分布
    assert json_count == len(
        json_data
    ), f"JSON数据数量不匹配: 期望{len(json_data)}, 实际{json_count}"
    assert pickle_count == len(
        pickle_data
    ), f"Pickle数据数量不匹配: 期望{len(pickle_data)}, 实际{pickle_count}"

    logger.info(
        "数据类型验证完成: JSON数据 %d 条, Pickle数据 %d 条", json_count, pickle_count
    )

    # 9. 测试时间戳范围查询对混合数据的支持
    # 获取前半部分数据（主要是JSON数据）
    mid_timestamp = base_timestamp + (total_count // 2) * 1000
    first_half = await cache.get_by_timestamp_range(
        test_key, end_timestamp=mid_timestamp
    )
    assert len(first_half) >= len(json_data) // 2, "时间戳范围查询对混合数据支持异常"

    # 获取后半部分数据（主要是Pickle数据）
    second_half = await cache.get_by_timestamp_range(
        test_key, start_timestamp=mid_timestamp
    )
    assert len(second_half) >= len(pickle_data), "时间戳范围查询对Pickle数据支持异常"

    logger.info(
        "时间戳范围查询混合数据测试通过: 前半部分 %d 条, 后半部分 %d 条",
        len(first_half),
        len(second_half),
    )

    # 10. 清理测试数据
    success = await cache.clear_queue(test_key)
    assert success, "清理测试数据失败"

    logger.info("✅ JSON和Pickle混合数据处理测试通过")


async def test_pickle_error_handling():
    """测试Pickle序列化的错误处理"""
    logger.info("开始测试Pickle序列化错误处理...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(max_length=10, expire_minutes=5)

    test_key = "test_pickle_error_handling"

    # 1. 清空测试队列
    await cache.clear_queue(test_key)

    # 2. 测试正常的Pickle数据
    normal_pickle_data = NonSerializableTestClass("normal", 42)
    success = await cache.append(test_key, normal_pickle_data)
    assert success, "正常Pickle数据添加失败"

    # 3. 获取并验证正常数据
    data_list = await cache.get_by_timestamp_range(test_key)
    assert len(data_list) == 1, "正常Pickle数据获取失败"

    retrieved = data_list[0]["data"]
    assert isinstance(retrieved, NonSerializableTestClass), "Pickle数据类型错误"
    assert retrieved.name == "normal" and retrieved.value == 42, "Pickle数据内容错误"
    assert retrieved.get_doubled_value() == 84, "Pickle对象功能异常"

    logger.info("正常Pickle数据处理验证通过")

    # 4. 测试混合数据的错误恢复
    mixed_data = [
        {"json_data": "this_is_json"},  # JSON数据
        NonSerializableTestClass("pickle1", 100),  # Pickle数据
        "simple_string",  # 简单字符串
        NonSerializableTestClass("pickle2", 200),  # 另一个Pickle数据
    ]

    for i, data in enumerate(mixed_data):
        success = await cache.append(test_key, data)
        assert success, f"混合数据第{i+1}条添加失败"

    # 5. 验证混合数据都能正确处理
    all_data = await cache.get_by_timestamp_range(test_key)
    assert (
        len(all_data) == 5
    ), f"混合数据总量错误: 期望5, 实际{len(all_data)}"  # 1个原有 + 4个新增

    # 统计各种数据类型
    json_count = sum(
        1
        for item in all_data
        if isinstance(item["data"], (dict, str))
        and not isinstance(item["data"], NonSerializableTestClass)
    )
    pickle_count = sum(
        1 for item in all_data if isinstance(item["data"], NonSerializableTestClass)
    )

    assert json_count >= 2, f"JSON数据数量异常: {json_count}"  # 至少包含dict和string
    assert (
        pickle_count >= 3
    ), f"Pickle数据数量异常: {pickle_count}"  # 3个NonSerializableTestClass对象

    logger.info(
        "混合数据错误处理测试通过: JSON类型 %d 条, Pickle类型 %d 条",
        json_count,
        pickle_count,
    )

    # 6. 清理测试数据
    success = await cache.clear_queue(test_key)
    assert success, "清理测试数据失败"

    logger.info("✅ Pickle序列化错误处理测试通过")


async def test_stress_operations():
    """测试压力操作：大量数据处理和清理"""
    logger.info("开始压力测试...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=100, expire_minutes=10, cleanup_probability=0.1
    )

    test_key = "test_length_cache_stress"

    # 1. 清空测试队列
    await cache.clear_queue(test_key)

    # 2. 批量追加数据（超过最大长度）
    batch_size = 500
    logger.info("追加 %d 条数据（超过最大长度100）...", batch_size)

    start_time = time.time()
    for i in range(batch_size):
        data = {
            "index": i,
            "timestamp": time.time(),
            "data": f"stress_test_data_{i}",
            "batch": "stress_test",
        }
        # 使用递增的时间戳确保顺序
        timestamp = int(time.time() * 1000) + i * 10
        success = await cache.append(test_key, data, timestamp=timestamp)
        assert success, f"批量追加第 {i+1} 条数据失败"

        # 每100条检查一次大小
        if (i + 1) % 100 == 0:
            current_size = await cache.get_queue_size(test_key)
            logger.info("已追加 %d 条，当前队列大小: %d", i + 1, current_size)

    elapsed = time.time() - start_time
    logger.info("数据追加完成，耗时: %.2f秒", elapsed)

    # 3. 验证队列大小（允许一定的超出，因为随机清理的概率性）
    final_size = await cache.get_queue_size(test_key)
    logger.info("压力测试后队列大小: %d", final_size)

    # 如果队列大小超出较多，手动触发清理来验证清理机制
    if final_size > 120:  # 允许一定的超出范围
        logger.info("队列大小超出较多，手动触发清理测试...")
        cleaned_count = await cache.cleanup_excess(test_key)
        logger.info("手动清理完成，清理了 %d 条数据", cleaned_count)

        # 验证清理后的大小
        size_after_cleanup = await cache.get_queue_size(test_key)
        assert (
            size_after_cleanup <= 100
        ), f"手动清理后队列大小仍超过限制: {size_after_cleanup}"
        logger.info("手动清理后队列大小: %d", size_after_cleanup)

    # 4. 获取最终队列大小（可能经过手动清理）
    current_size = await cache.get_queue_size(test_key)

    # 验证队列统计信息
    stats = await cache.get_queue_stats(test_key)
    assert stats["total_count"] == current_size, "统计信息与实际大小不匹配"
    assert stats["is_full"] == (current_size >= 100), "队列满状态判断错误"

    # 5. 再次手动清理验证机制稳定性
    logger.info("再次手动清理队列验证机制稳定性...")
    additional_cleaned = await cache.cleanup_excess(test_key)
    logger.info("额外清理完成，清理数量: %d", additional_cleaned)

    # 6. 清理测试数据
    success = await cache.clear_queue(test_key)
    assert success, "清理测试数据失败"

    logger.info("✅ 压力测试通过")


async def test_expiry_mechanism():
    """测试过期机制"""
    logger.info("开始测试过期机制...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(
        max_length=10, expire_minutes=1
    )  # 1分钟过期

    test_key = "test_length_cache_expiry"

    # 1. 清空测试队列
    await cache.clear_queue(test_key)

    # 2. 添加测试数据
    test_data = {"test": "expiry", "timestamp": time.time()}
    success = await cache.append(test_key, test_data)
    assert success, "添加测试数据失败"

    # 3. 验证数据存在
    size = await cache.get_queue_size(test_key)
    assert size == 1, "数据未正确添加"

    stats = await cache.get_queue_stats(test_key)
    assert stats["ttl_seconds"] > 0, "TTL应该大于0"
    logger.info("数据TTL: %d 秒", stats["ttl_seconds"])

    # 4. 等待过期
    logger.info("等待数据过期（70秒）...")
    await asyncio.sleep(70)  # 等待超过1分钟

    # 5. 验证数据已过期
    expired_size = await cache.get_queue_size(test_key)
    assert expired_size == 0, f"数据未正确过期，队列大小: {expired_size}"

    expired_stats = await cache.get_queue_stats(test_key)
    assert expired_stats["total_count"] == 0, "过期后统计信息应为0"

    logger.info("✅ 过期机制测试通过")


async def test_compatibility_layer():
    """测试向后兼容层"""
    logger.info("开始测试向后兼容层...")

    # 获取默认管理器实例
    default_manager = get_bean("redis_length_cache_manager")
    test_key = "test_length_cache_compat"

    # 1. 清空测试队列
    await default_manager.clear_queue(test_key)

    # 2. 测试基本操作
    test_data = {"test": "compatibility", "type": "default_manager"}
    success = await default_manager.append(test_key, test_data)
    assert success, "向后兼容层追加数据失败"

    size = await default_manager.get_queue_size(test_key)
    assert size == 1, "向后兼容层队列大小错误"

    stats = await default_manager.get_queue_stats(test_key)
    assert stats["total_count"] == 1, "向后兼容层统计信息错误"

    # 3. 测试带时间戳的追加
    dt = datetime.now()
    success = await default_manager.append(test_key, "datetime_test", timestamp=dt)
    assert success, "向后兼容层datetime追加失败"

    final_size = await default_manager.get_queue_size(test_key)
    assert final_size == 2, "向后兼容层最终大小错误"

    # 4. 测试新增的时间戳范围查询方法
    # 添加一些测试数据
    base_timestamp = int(time.time() * 1000)
    for i in range(3):
        timestamp = base_timestamp + i * 5000
        data = {"index": i, "content": f"compat_data_{i}"}
        success = await default_manager.append(test_key, data, timestamp=timestamp)
        assert success, f"向后兼容层添加测试数据{i}失败"

    # 测试按时间戳范围获取
    range_data = await default_manager.get_by_timestamp_range(test_key)
    assert (
        len(range_data) == 5
    ), f"向后兼容层时间戳范围查询失败，期望5条，实际{len(range_data)}条"  # 2条原有数据 + 3条新数据

    # 测试限制数量
    limited_data = await default_manager.get_by_timestamp_range(test_key, limit=2)
    assert len(limited_data) == 2, "向后兼容层限制数量功能失败"

    logger.info("向后兼容层新增方法测试通过")

    # 5. 清理测试数据
    success = await default_manager.clear_queue(test_key)
    assert success, "向后兼容层清理数据失败"

    logger.info("✅ 向后兼容层测试通过")


async def main():
    """主测试函数"""
    logger.info("=" * 50)
    logger.info("Redis 长度限制缓存管理器测试开始")
    logger.info("=" * 50)

    try:
        # 运行所有测试
        await test_json_pickle_mixed_data()
        await test_basic_operations()
        await test_length_cleanup()
        await test_timestamp_compatibility()
        await test_timestamp_range_query()
        await test_pickle_error_handling()
        await test_stress_operations()
        await test_expiry_mechanism()
        await test_compatibility_layer()

        logger.info("=" * 50)
        logger.info("✅ 所有测试通过")
        logger.info("=" * 50)

    except AssertionError as e:
        logger.error("❌ 测试失败: %s", str(e))
        raise
    except Exception as e:
        logger.error("❌ 测试出错: %s", str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
