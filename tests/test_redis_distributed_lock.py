"""
Redis分布式锁测试

测试场景包括：
1. 基本的锁获取和释放
2. 锁的可重入性
3. 超时机制
4. 并发竞争
5. 装饰器使用
"""

import asyncio
from core.lock.redis_distributed_lock import with_distributed_lock, distributed_lock


async def test_basic_lock_operations(redis_distributed_lock_manager):
    """测试基本的锁操作"""
    resource = "test_resource"
    lock = redis_distributed_lock_manager.get_lock(resource)

    # 测试获取锁
    async with lock.acquire() as acquired:
        assert acquired, "应该成功获取锁"
        assert await lock.is_locked(), "资源应该处于锁定状态"
        assert await lock.is_owned_by_current_coroutine(), "锁应该被当前协程持有"

    # 测试锁释放
    assert not await lock.is_locked(), "锁应该已被释放"
    assert not await lock.is_owned_by_current_coroutine(), "锁不应该被当前协程持有"


async def test_lock_reentrant(redis_distributed_lock_manager):
    """测试锁的可重入性"""
    resource = "test_reentrant"
    lock = redis_distributed_lock_manager.get_lock(resource)

    async with lock.acquire() as acquired:
        assert acquired, "第一次应该成功获取锁"
        count1 = await lock.get_reentry_count()
        assert count1 == 1, "第一次获取锁，重入计数应该为1"

        # 重入获取锁
        async with lock.acquire() as reacquired:
            assert reacquired, "第二次应该成功获取锁（重入）"
            count2 = await lock.get_reentry_count()
            assert count2 == 2, "第二次获取锁，重入计数应该为2"

        # 释放一次后，锁应该还在
        count3 = await lock.get_reentry_count()
        assert count3 == 1, "释放一次后，重入计数应该为1"
        assert await lock.is_locked(), "释放一次后，锁应该还在"

    # 完全释放后，锁应该消失
    assert not await lock.is_locked(), "完全释放后，锁应该消失"
    assert await lock.get_reentry_count() == 0, "完全释放后，重入计数应该为0"


async def test_lock_expiration(redis_distributed_lock_manager):
    """测试锁的过期机制"""
    resource = "test_expiration"

    # 测试场景1：基本过期
    lock1 = redis_distributed_lock_manager.get_lock(resource)
    async with lock1.acquire(timeout=1) as acquired:  # 1秒过期
        assert acquired, "应该成功获取锁"
        assert await lock1.is_locked(), "资源应该处于锁定状态"
        assert await lock1.is_owned_by_current_coroutine(), "锁应该被当前协程持有"

        # 等待不到过期时间，锁应该还在
        await asyncio.sleep(0.5)
        assert await lock1.is_locked(), "未到过期时间，锁应该还在"

        # 等待过期
        await asyncio.sleep(1)
        assert not await lock1.is_locked(), "锁应该已经过期释放"

    # 测试场景2：过期后其他协程可以获取锁
    async def try_acquire_expired_lock():
        lock2 = redis_distributed_lock_manager.get_lock(resource)
        async with lock2.acquire(timeout=5) as acquired:  # 设置足够长的过期时间
            assert acquired, "在原锁过期后，新的协程应该能获取锁"
            assert await lock2.is_locked(), "新获取的锁应该处于锁定状态"
            assert (
                await lock2.is_owned_by_current_coroutine()
            ), "新的锁应该被当前协程持有"
            return True
        return False

    success = await try_acquire_expired_lock()
    assert success, "应该能在原锁过期后获取新锁"

    # 测试场景3：不同过期时间
    test_times = [0.5, 1, 2]  # 测试不同的过期时间
    for expire_time in test_times:
        lock = redis_distributed_lock_manager.get_lock(f"{resource}_{expire_time}")
        async with lock.acquire(timeout=expire_time) as acquired:
            assert acquired, f"应该成功获取锁（过期时间：{expire_time}秒）"

            # 等待一半时间，锁应该还在
            await asyncio.sleep(expire_time / 2)
            assert (
                await lock.is_locked()
            ), f"过期时间{expire_time}秒，{expire_time/2}秒后锁应该还在"

            # 等待剩余时间+一点余量，锁应该过期
            await asyncio.sleep(expire_time / 2 + 0.1)
            assert (
                not await lock.is_locked()
            ), f"过期时间{expire_time}秒，{expire_time+0.1}秒后锁应该已过期"

    # 测试场景4：重入时更新过期时间
    lock3 = redis_distributed_lock_manager.get_lock("test_reentry_expiration")
    async with lock3.acquire(timeout=1) as acquired1:  # 1秒过期
        assert acquired1, "第一次应该成功获取锁"

        # 等待0.8秒（接近过期）
        await asyncio.sleep(0.8)
        assert await lock3.is_locked(), "接近过期时锁应该还在"

        # 重入并设置新的过期时间
        async with lock3.acquire(timeout=2) as acquired2:  # 2秒过期
            assert acquired2, "应该能重入获取锁"
            assert await lock3.get_reentry_count() == 2, "重入计数应该为2"

            # 等待1.2秒（超过原来的1秒过期时间）
            await asyncio.sleep(1.2)
            assert await lock3.is_locked(), "重入后设置了新的过期时间，锁应该还在"
            assert await lock3.get_reentry_count() == 2, "重入计数应该保持为2"


async def test_concurrent_lock_competition(redis_distributed_lock_manager):
    """测试并发竞争场景"""
    resource = "test_concurrent"
    results = []

    async def compete_for_lock(task_id):
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire(blocking_timeout=1) as acquired:
            if acquired:
                results.append(task_id)
                await asyncio.sleep(0.1)  # 模拟工作负载

    # 创建多个并发任务
    tasks = [compete_for_lock(i) for i in range(5)]
    await asyncio.gather(*tasks)

    # 验证结果
    assert len(results) > 0, "至少有一个任务应该获取到锁"
    assert len(results) == len(set(results)), "每个任务ID应该只出现一次"


@with_distributed_lock("test_decorator")
async def decorated_function(value):
    return value * 2


@with_distributed_lock("test_decorator_{value}")
async def decorated_function_with_format(value):
    return value * 2


async def test_lock_decorator(_redis_distributed_lock_manager):
    """测试装饰器功能"""
    # 测试基本装饰器
    result1 = await decorated_function(21)
    assert result1 == 42, "装饰器不应该影响函数返回值"

    # 测试带格式化字符串的装饰器
    result2 = await decorated_function_with_format(21)
    assert result2 == 42, "带格式化字符串的装饰器不应该影响函数返回值"


async def test_force_unlock(redis_distributed_lock_manager):
    """测试强制解锁功能"""
    resource = "test_force_unlock"
    lock = redis_distributed_lock_manager.get_lock(resource)

    async with lock.acquire() as acquired:
        assert acquired, "应该成功获取锁"

        # 强制解锁
        success = await redis_distributed_lock_manager.force_unlock(resource)
        assert success, "强制解锁应该成功"
        assert not await lock.is_locked(), "强制解锁后锁应该被释放"


async def test_blocking_timeout_and_reentry(redis_distributed_lock_manager):
    """测试阻塞超时和可重入性（使用asyncio任务）"""
    resource = "test_blocking"

    # 测试场景1：同一任务的可重入
    async def reentry_test():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire() as acquired1:
            assert acquired1, "第一次获取锁应该成功"
            assert await lock.get_reentry_count() == 1, "第一次获取后重入计数应该为1"

            # 同一任务重入获取
            async with lock.acquire() as acquired2:
                assert acquired2, "同一任务重入应该成功"
                assert await lock.get_reentry_count() == 2, "重入后计数应该为2"

                # 再次重入
                async with lock.acquire() as acquired3:
                    assert acquired3, "同一任务第三次重入应该成功"
                    assert (
                        await lock.get_reentry_count() == 3
                    ), "第三次重入后计数应该为3"
                    await asyncio.sleep(0.1)  # 确保任务切换

    # 创建并运行任务
    task1 = asyncio.create_task(reentry_test())
    await task1

    # 测试场景2：不同任务间的阻塞和重入
    async def blocking_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire() as acquired:
            assert acquired, "第一个任务应该能获取锁"
            assert await lock.get_reentry_count() == 1, "第一个任务的重入计数应该为1"
            await asyncio.sleep(1)  # 持有锁一段时间

    async def competing_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire(blocking_timeout=0.5) as acquired:
            assert not acquired, "第二个任务不应该能获取到锁"
            assert await lock.get_reentry_count() == 0, "获取失败时重入计数应该为0"

    # 创建两个竞争的任务
    task2 = asyncio.create_task(blocking_task())
    await asyncio.sleep(0.1)  # 确保第一个任务先获取锁
    task3 = asyncio.create_task(competing_task())

    # 等待任务完成
    await asyncio.gather(task2, task3)

    # 测试场景3：任务嵌套时的可重入性
    async def parent_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire() as acquired:
            assert acquired, "父任务应该能获取锁"
            assert await lock.get_reentry_count() == 1, "父任务重入计数应该为1"

            # 创建子任务
            child = asyncio.create_task(child_task())
            await asyncio.sleep(0.1)  # 确保子任务有机会运行

            # 父任务重入
            async with lock.acquire() as reentry:
                assert reentry, "父任务重入应该成功"
                assert await lock.get_reentry_count() == 2, "父任务重入后计数应该为2"
                await asyncio.sleep(0.1)  # 再次给子任务机会

            await child  # 等待子任务完成

    async def child_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire(blocking_timeout=0.5) as acquired:
            assert not acquired, "子任务不应该能获取到父任务持有的锁"
            assert await lock.get_reentry_count() == 0, "子任务获取失败时计数应该为0"

    # 运行父子任务测试
    await parent_task()

    # 测试场景4：锁释放后新任务的获取
    async def final_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire(blocking_timeout=0.5) as acquired:
            assert acquired, "锁释放后新任务应该能获取到锁"
            assert await lock.get_reentry_count() == 1, "新任务获取锁后计数应该为1"

    # 确保锁已释放
    assert not await redis_distributed_lock_manager.get_lock(
        resource
    ).is_locked(), "所有任务完成后锁应该被释放"

    # 运行最终测试
    final = asyncio.create_task(final_task())
    await final


async def run_all_tests():
    """运行所有测试"""
    from core.di.utils import get_bean_by_type
    from core.lock.redis_distributed_lock import RedisDistributedLockManager

    print("开始运行Redis分布式锁测试...")

    # 获取锁管理器实例
    lock_manager = get_bean_by_type(RedisDistributedLockManager)

    # 定义所有测试函数
    tests = [
        test_basic_lock_operations,
        test_lock_reentrant,
        test_lock_expiration,  # 新的过期测试
        test_concurrent_lock_competition,
        test_lock_decorator,
        test_force_unlock,
        test_blocking_timeout_and_reentry,  # 更新后的阻塞和可重入测试
        test_convenient_context_manager,
        test_context_manager_with_timeout,
        test_context_manager_concurrent,
    ]

    # 运行所有测试
    for test_func in tests:
        print(f"\n运行测试: {test_func.__name__}")
        print("-" * 50)
        try:
            await test_func(lock_manager)
            print(f"✅ {test_func.__name__} 测试通过")
        except AssertionError as e:
            print(f"❌ {test_func.__name__} 测试失败: {str(e)}")
        except (ConnectionError, TimeoutError, OSError) as e:
            print(f"❌ {test_func.__name__} 测试出错: {str(e)}")

    print("\n测试完成!")


async def test_convenient_context_manager(_redis_distributed_lock_manager):
    """测试便捷的上下文管理器函数"""
    resource = "test_context_manager"

    # 测试基本使用
    async with distributed_lock(resource) as acquired:
        assert acquired, "应该成功获取锁"

        # 测试可重入性
        async with distributed_lock(resource) as reacquired:
            assert reacquired, "应该支持可重入"

    # 测试不同资源的锁
    async with distributed_lock("resource1") as acquired1:
        assert acquired1, "应该成功获取resource1的锁"

        async with distributed_lock("resource2") as acquired2:
            assert acquired2, "应该成功获取resource2的锁"

    print("✅ 便捷上下文管理器测试通过")


async def test_context_manager_with_timeout(_redis_distributed_lock_manager):
    """测试上下文管理器的超时功能"""
    resource = "test_timeout_context"

    # 测试自定义超时参数
    async with distributed_lock(
        resource, timeout=30.0, blocking_timeout=5.0
    ) as acquired:
        assert acquired, "应该成功获取锁"

    print("✅ 上下文管理器超时测试通过")


async def test_context_manager_concurrent(_redis_distributed_lock_manager):
    """测试上下文管理器的并发情况"""
    resource = "test_concurrent_context"
    results = []

    async def worker(worker_id: int):
        async with distributed_lock(resource, blocking_timeout=0.2) as acquired:
            if acquired:
                results.append(f"worker_{worker_id}")
                # 持有锁足够长的时间，确保其他worker超时
                await asyncio.sleep(0.5)

    # 真正并发执行多个worker
    tasks = [worker(i) for i in range(3)]
    await asyncio.gather(*tasks)

    # 由于锁的存在和短暂的blocking_timeout，应该只有一个worker成功
    assert len(results) == 1, f"应该只有一个worker成功，但得到: {results}"
    print(f"✅ 上下文管理器并发测试通过，成功的worker: {results[0]}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
