"""
Pickle 序列化大小分析测试

测试各种类型和大小的对象在 Pickle 序列化后的二进制大小
包括：
1. 基础数据类型大小分析
2. 复杂对象大小分析
3. 大型数据结构大小分析
4. 函数和类对象大小分析
5. 嵌套结构大小分析
"""

import asyncio
import pickle
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from core.di.utils import get_bean
from core.observation.logger import get_logger
from core.cache.redis_cache_queue.redis_data_processor import RedisDataProcessor

logger = get_logger(__name__)


def format_size(size_bytes: int) -> str:
    """格式化字节大小为可读格式"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


class ComplexTestObject:
    """复杂测试对象"""

    def __init__(self, name: str, data_size: int):
        self.name = name
        self.created_at = datetime.now()
        self.id = id(self)
        self.data = list(range(data_size))
        self.metadata = {
            "version": "1.0",
            "type": "test_object",
            "size": data_size,
            "nested": {
                "level1": {"level2": {"level3": "deep_data"}},
                "timestamps": [
                    datetime.now() + timedelta(seconds=i) for i in range(10)
                ],
            },
        }
        # 使用不能JSON序列化的数据类型
        self.complex_types = {
            "set_data": {1, 2, 3, 4, 5},
            "bytes_data": b"binary_data_example",
            "tuple_data": (1, 2, 3, 4, 5),
        }

    def multiply_value(self, value, multiplier=2):
        return value * multiplier

    def power_value(self, value, power=2):
        return value**power

    def custom_process(self, value):
        return f"processed_{value}_{self.name}_{len(self.data)}"

    def get_summary(self):
        return f"ComplexTestObject(name={self.name}, data_len={len(self.data)})"


class LargeDataContainer:
    """大型数据容器"""

    def __init__(self, size_mb: float):
        # 创建指定大小的数据
        target_size = int(size_mb * 1024 * 1024 / 8)  # 假设每个数字8字节
        self.large_list = list(range(target_size))
        self.large_dict = {
            f"key_{i}": f"value_{i}_{'x' * 100}" for i in range(target_size // 100)
        }
        self.metadata = {
            "size_mb": size_mb,
            "created": datetime.now(),
            "type": "large_container",
        }


async def test_basic_types_size():
    """测试基础数据类型的序列化大小"""
    logger.info("开始测试基础数据类型序列化大小...")

    test_data = {
        "空字符串": "",
        "短字符串": "hello",
        "中等字符串": "x" * 100,
        "长字符串": "x" * 1000,
        "超长字符串": "x" * 10000,
        "小整数": 42,
        "大整数": 123456789012345,
        "浮点数": 3.14159265359,
        "布尔值True": True,
        "布尔值False": False,
        "None": None,
        "空列表": [],
        "小列表": [1, 2, 3],
        "中等列表": list(range(100)),
        "大列表": list(range(1000)),
        "空字典": {},
        "小字典": {"a": 1, "b": 2, "c": 3},
        "中等字典": {f"key_{i}": i for i in range(100)},
        "大字典": {f"key_{i}": f"value_{i}" for i in range(1000)},
    }

    logger.info("=" * 60)
    logger.info("基础数据类型 Pickle 序列化大小分析")
    logger.info("=" * 60)

    for name, data in test_data.items():
        # JSON序列化大小（如果可能）
        json_size = "N/A"
        try:
            import json

            json_data = json.dumps(data, ensure_ascii=False)
            json_size = format_size(len(json_data.encode('utf-8')))
        except (TypeError, ValueError):
            json_size = "无法JSON序列化"

        # Pickle序列化大小
        pickle_data = pickle.dumps(data)
        pickle_size = format_size(len(pickle_data))

        # 使用RedisDataProcessor处理
        processed_data = RedisDataProcessor.serialize_data(data)
        if isinstance(processed_data, bytes):
            processed_size = format_size(len(processed_data))
            serialization_type = "Pickle"
        else:
            processed_size = format_size(len(processed_data.encode('utf-8')))
            serialization_type = "JSON"

        logger.info(
            "%-15s | JSON: %-12s | Pickle: %-12s | 处理器: %-12s (%s)",
            name,
            json_size,
            pickle_size,
            processed_size,
            serialization_type,
        )

    logger.info("✅ 基础数据类型大小分析完成")


async def test_complex_objects_size():
    """测试复杂对象的序列化大小"""
    logger.info("开始测试复杂对象序列化大小...")

    test_objects = [
        ("小型复杂对象", ComplexTestObject("small", 10)),
        ("中型复杂对象", ComplexTestObject("medium", 100)),
        ("大型复杂对象", ComplexTestObject("large", 1000)),
        ("超大复杂对象", ComplexTestObject("xlarge", 10000)),
    ]

    logger.info("=" * 60)
    logger.info("复杂对象 Pickle 序列化大小分析")
    logger.info("=" * 60)

    for name, obj in test_objects:
        # Pickle序列化
        pickle_data = pickle.dumps(obj)
        pickle_size = format_size(len(pickle_data))

        # 使用RedisDataProcessor处理
        processed_data = RedisDataProcessor.serialize_data(obj)
        processed_size = format_size(len(processed_data))

        # 对象内存大小估算
        obj_memory = format_size(
            sys.getsizeof(obj) + sys.getsizeof(obj.data) + sys.getsizeof(obj.metadata)
        )

        logger.info(
            "%-15s | 内存: %-12s | Pickle: %-12s | 处理器: %-12s",
            name,
            obj_memory,
            pickle_size,
            processed_size,
        )

    logger.info("✅ 复杂对象大小分析完成")


async def test_large_data_structures():
    """测试大型数据结构的序列化大小"""
    logger.info("开始测试大型数据结构序列化大小...")

    logger.info("=" * 60)
    logger.info("大型数据结构 Pickle 序列化大小分析")
    logger.info("=" * 60)

    # 测试不同大小的数据结构
    sizes = [0.1, 0.5, 1.0, 2.0, 5.0]  # MB

    for size_mb in sizes:
        logger.info(f"测试 {size_mb} MB 数据容器...")

        try:
            # 创建大型数据容器
            container = LargeDataContainer(size_mb)

            # Pickle序列化
            start_time = time.time()
            pickle_data = pickle.dumps(container)
            pickle_time = time.time() - start_time
            pickle_size = format_size(len(pickle_data))

            # 使用RedisDataProcessor处理
            start_time = time.time()
            processed_data = RedisDataProcessor.serialize_data(container)
            process_time = time.time() - start_time
            processed_size = format_size(len(processed_data))

            # 压缩率计算
            original_estimate = size_mb * 1024 * 1024
            compression_ratio = len(pickle_data) / original_estimate

            logger.info(
                "%-8s MB | Pickle: %-12s (%.2fs) | 处理器: %-12s (%.2fs) | 压缩率: %.2f",
                f"{size_mb:.1f}",
                pickle_size,
                pickle_time,
                processed_size,
                process_time,
                compression_ratio,
            )

        except MemoryError:
            logger.warning("%-8s MB | 内存不足，跳过测试", f"{size_mb:.1f}")
        except Exception as e:
            logger.error("%-8s MB | 测试失败: %s", f"{size_mb:.1f}", str(e))

    logger.info("✅ 大型数据结构大小分析完成")


async def test_function_and_class_objects():
    """测试函数和类对象的序列化大小"""
    logger.info("开始测试函数和类对象序列化大小...")

    # 各种函数和类对象
    def simple_function(x):
        return x * 2

    def complex_function(x, y, z=10):
        """复杂函数with文档字符串"""
        result = x + y + z
        for i in range(100):
            result += i
        return result

    class SimpleClass:
        def __init__(self, value):
            self.value = value

        def method(self):
            return self.value * 2

    class ComplexClass:
        """复杂类with多个方法"""

        class_var = "shared_data"

        def __init__(self, name, data):
            self.name = name
            self.data = data
            self.timestamp = datetime.now()

        def method1(self):
            return len(self.data)

        def method2(self, multiplier=2):
            return [x * multiplier for x in self.data]

        @staticmethod
        def static_method():
            return "static_result"

        @classmethod
        def class_method(cls):
            return cls.class_var

    test_objects = [
        ("简单函数", simple_function),
        ("复杂函数", complex_function),
        ("简单类实例", SimpleClass(42)),
        ("复杂类实例", ComplexClass("test", list(range(100)))),
        (
            "包含set的字典",
            {
                "set_data": {1, 2, 3, 4, 5},
                "bytes_data": b"function_test_binary",
                "tuple_data": (1, 2, 3),
            },
        ),
        (
            "混合对象",
            {
                "functions": [simple_function, complex_function],
                "objects": [SimpleClass(i) for i in range(10)],
                "data": list(range(1000)),
                "complex_types": {
                    "set_data": {10, 20, 30},
                    "bytes_data": b"mixed_binary_data",
                },
                "metadata": {"type": "mixed", "count": 10},
            },
        ),
    ]

    logger.info("=" * 60)
    logger.info("函数和类对象 Pickle 序列化大小分析")
    logger.info("=" * 60)

    for name, obj in test_objects:
        try:
            # Pickle序列化
            pickle_data = pickle.dumps(obj)
            pickle_size = format_size(len(pickle_data))

            # 使用RedisDataProcessor处理
            processed_data = RedisDataProcessor.serialize_data(obj)
            processed_size = format_size(len(processed_data))

            # 对象内存大小
            obj_memory = format_size(sys.getsizeof(obj))

            logger.info(
                "%-15s | 内存: %-12s | Pickle: %-12s | 处理器: %-12s",
                name,
                obj_memory,
                pickle_size,
                processed_size,
            )

        except Exception as e:
            logger.error("%-15s | 序列化失败: %s", name, str(e))

    logger.info("✅ 函数和类对象大小分析完成")


async def test_nested_structures():
    """测试嵌套结构的序列化大小"""
    logger.info("开始测试嵌套结构序列化大小...")

    # 创建不同深度的嵌套结构
    def create_nested_dict(depth: int, width: int = 3):
        """创建指定深度和宽度的嵌套字典"""
        if depth == 0:
            return f"leaf_value_{width}"

        return {f"key_{i}": create_nested_dict(depth - 1, width) for i in range(width)}

    def create_nested_list(depth: int, width: int = 3):
        """创建指定深度和宽度的嵌套列表"""
        if depth == 0:
            return f"leaf_{width}"

        return [create_nested_list(depth - 1, width) for _ in range(width)]

    test_structures = [
        ("嵌套字典-深度2", create_nested_dict(2, 3)),
        ("嵌套字典-深度3", create_nested_dict(3, 3)),
        ("嵌套字典-深度4", create_nested_dict(4, 2)),
        ("嵌套列表-深度2", create_nested_list(2, 3)),
        ("嵌套列表-深度3", create_nested_list(3, 3)),
        ("嵌套列表-深度4", create_nested_list(4, 2)),
        (
            "混合嵌套",
            {
                "dict_part": create_nested_dict(3, 2),
                "list_part": create_nested_list(3, 2),
                "objects": [ComplexTestObject(f"nested_{i}", 50) for i in range(5)],
                "complex_types": {
                    "nested_set": {frozenset({1, 2}), frozenset({3, 4})},
                    "nested_bytes": b"nested_binary_data",
                    "nested_tuple": ((1, 2), (3, 4), (5, 6)),
                },
            },
        ),
    ]

    logger.info("=" * 60)
    logger.info("嵌套结构 Pickle 序列化大小分析")
    logger.info("=" * 60)

    for name, structure in test_structures:
        try:
            # Pickle序列化
            start_time = time.time()
            pickle_data = pickle.dumps(structure)
            pickle_time = time.time() - start_time
            pickle_size = format_size(len(pickle_data))

            # 使用RedisDataProcessor处理
            start_time = time.time()
            processed_data = RedisDataProcessor.serialize_data(structure)
            process_time = time.time() - start_time
            processed_size = format_size(len(processed_data))

            logger.info(
                "%-20s | Pickle: %-12s (%.3fs) | 处理器: %-12s (%.3fs)",
                name,
                pickle_size,
                pickle_time,
                processed_size,
                process_time,
            )

        except Exception as e:
            logger.error("%-20s | 序列化失败: %s", name, str(e))

    logger.info("✅ 嵌套结构大小分析完成")


async def test_redis_storage_efficiency():
    """测试Redis存储效率"""
    logger.info("开始测试Redis存储效率...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(max_length=1000, expire_minutes=10)

    test_key = "pickle_size_test"
    await cache.clear_queue(test_key)

    # 测试数据
    test_data = [
        ("小JSON对象", {"name": "test", "value": 123}),
        ("大JSON对象", {"data": list(range(1000)), "metadata": {"type": "large"}}),
        ("小Pickle对象", ComplexTestObject("small", 10)),
        ("大Pickle对象", ComplexTestObject("large", 1000)),
    ]

    logger.info("=" * 60)
    logger.info("Redis 存储效率分析")
    logger.info("=" * 60)

    for name, data in test_data:
        # 序列化大小
        processed_data = RedisDataProcessor.process_data_for_storage(data)
        storage_size = (
            len(processed_data)
            if isinstance(processed_data, bytes)
            else len(processed_data.encode('utf-8'))
        )

        # 存储到Redis
        start_time = time.time()
        success = await cache.append(test_key, data)
        store_time = time.time() - start_time

        # 从Redis读取
        start_time = time.time()
        retrieved_data = await cache.get_by_timestamp_range(test_key, limit=1)
        read_time = time.time() - start_time

        if success and retrieved_data:
            logger.info(
                "%-15s | 存储: %-12s | 写入: %.3fs | 读取: %.3fs | 状态: ✅",
                name,
                format_size(storage_size),
                store_time,
                read_time,
            )
        else:
            logger.error(
                "%-15s | 存储: %-12s | 状态: ❌", name, format_size(storage_size)
            )

        # 清理单个测试数据
        await cache.clear_queue(test_key)

    logger.info("✅ Redis存储效率分析完成")


async def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("Pickle 序列化大小分析测试开始")
    logger.info("=" * 60)

    try:
        await test_basic_types_size()
        await test_complex_objects_size()
        await test_large_data_structures()
        await test_function_and_class_objects()
        await test_nested_structures()
        await test_redis_storage_efficiency()

        logger.info("=" * 60)
        logger.info("✅ 所有Pickle大小分析测试通过")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("❌ 测试过程中发生错误: %s", str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
