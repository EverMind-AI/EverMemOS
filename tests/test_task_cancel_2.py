import asyncio
import time


async def child_task():
    try:
        print(f"[{time.time():.2f}] 子任务开始")
        await asyncio.sleep(2)
        print(f"[{time.time():.2f}] 子任务完成")
    except asyncio.CancelledError:
        print(f"[{time.time():.2f}] 子任务收到取消")
        raise


async def parent_task():
    print(f"[{time.time():.2f}] 父任务开始")

    # 在 await 前执行一些工作
    print(f"[{time.time():.2f}] 父任务执行一些工作")
    await asyncio.sleep(5)  # 模拟工作

    try:
        print(f"[{time.time():.2f}] 父任务即将 await 子任务")
        # 关键点：在这里被取消
        await child_task()
        print(f"[{time.time():.2f}] 父任务完成等待")
    except asyncio.CancelledError:
        print(f"[{time.time():.2f}] 父任务收到取消")
        raise


async def main():
    parent = asyncio.create_task(parent_task())

    # 立即取消父任务（在它到达 await 前）
    print(f"[{time.time():.2f}] 立即发送取消请求")
    await asyncio.sleep(1)
    parent.cancel()

    try:
        await parent
    except asyncio.CancelledError:
        print(f"[{time.time():.2f}] 主程序捕获取消")


asyncio.run(main())
