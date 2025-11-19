import asyncio
import time


async def child_task(name, swallow_cancel=False, hang=False):
    try:
        print(f"[{name}] 子任务开始运行")
        await asyncio.sleep(5)  # 模拟长时间工作
        print(f"[{name}] 子任务正常完成")
    except asyncio.CancelledError:
        print(f"[{name}] 子任务收到取消请求")
        if hang:
            print(f"[{name}] 子任务故意挂起不退出")
            while True:  # 模拟卡死
                print(f"[{name}] 子任务卡死挂起中")
                await asyncio.sleep(1)
        elif swallow_cancel:
            print(f"[{name}] 子任务吞掉取消请求")
        else:
            print(f"[{name}] 子任务重新抛出取消请求")
            raise  # 重新抛出异常

    print(f"[{name}] 子任务最终结束")


async def parent_task(child_name, swallow=False, hang=False):
    try:
        print(f"[父任务] 开始运行，创建子任务 {child_name}")
        child = asyncio.create_task(child_task(child_name, swallow, hang))

        # 模拟父任务的其他工作
        await asyncio.sleep(2)
        print(f"[父任务] 等待子任务完成")
        await child
        print(f"[父任务] 正常完成")
    except asyncio.CancelledError:
        print(f"[父任务] 收到取消请求")
        raise  # 重新抛出异常
    print(f"[父任务] 最终结束")


async def main():
    # 场景1：正常取消传播
    print("\n===== 场景1：正常取消传播 =====")
    task1 = asyncio.create_task(parent_task("正常子任务"))
    await asyncio.sleep(1)
    task1.cancel()
    try:
        await task1
    except asyncio.CancelledError:
        print("[主程序] 父任务已取消")
    await asyncio.sleep(5)
    # 场景2：子任务吞掉取消
    print("\n===== 场景2：子任务吞掉取消 =====")
    task2 = asyncio.create_task(parent_task("吞取消子任务", swallow=True))
    await asyncio.sleep(1)
    task2.cancel()
    try:
        await task2
    except asyncio.CancelledError:
        print("[主程序] 父任务已取消")

    await asyncio.sleep(5)
    # 场景3：子任务卡死
    print("\n===== 场景3：子任务卡死 =====")
    task3 = asyncio.create_task(parent_task("卡死子任务", hang=True))
    await asyncio.sleep(1)
    task3.cancel()
    try:
        await task3
    except asyncio.CancelledError:
        print("[主程序] 父任务已取消")
    print("[主程序] 最终结束")


if __name__ == "__main__":
    asyncio.run(main())
