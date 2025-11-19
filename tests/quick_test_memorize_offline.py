#!/usr/bin/env python3
"""
memorize_offline 离线记忆处理测试脚本

使用方法:
    python tests/test_memorize_offline.py           # 默认测试最近7天
    python tests/test_memorize_offline.py 3         # 测试最近3天
    python tests/test_memorize_offline.py 1 debug   # 测试最近1天，详细输出
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

from memory_layer.memory_manager import MemorizeOfflineRequest
from biz_layer.tanka_memorize import memorize_offline
from common_utils.datetime_utils import get_now_with_timezone


async def test_memorize_offline(days=7, debug=False):
    """测试memorize_offline流程"""
    print(f"🚀 测试memorize_offline流程（最近{days}天）")

    # 创建测试时间范围
    current_time = get_now_with_timezone()
    request = MemorizeOfflineRequest(
        memorize_from=current_time - timedelta(days=days), memorize_to=current_time
    )

    print(
        f"⏰ 时间范围: {request.memorize_from.strftime('%Y-%m-%d %H:%M')} ~ {request.memorize_to.strftime('%Y-%m-%d %H:%M')}"
    )

    try:
        # 执行测试
        start_time = datetime.now()
        result = await memorize_offline(request)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        print(f"✅ 测试完成! 耗时: {duration:.2f}秒")
        print(f"📊 提取记忆: {len(result) if result else 0} 个")

        if result and len(result) > 0:
            # 统计记忆类型
            type_stats = {}
            group_stats = {}

            for memory in result:
                # 记忆类型统计
                memory_type = (
                    memory.memory_type.value
                    if hasattr(memory.memory_type, 'value')
                    else str(memory.memory_type)
                )
                type_stats[memory_type] = type_stats.get(memory_type, 0) + 1

                # 群组统计
                if hasattr(memory, 'group_id') and memory.group_id:
                    group_stats[memory.group_id] = (
                        group_stats.get(memory.group_id, 0) + 1
                    )

            print("📈 记忆类型分布:")
            for memory_type, count in type_stats.items():
                print(f"   {memory_type}: {count} 个")

            if debug and group_stats:
                print("👥 群组分布:")
                for group_id, count in list(group_stats.items())[:5]:  # 只显示前5个
                    print(f"   {group_id}: {count} 个")
                if len(group_stats) > 5:
                    print(f"   ... 还有 {len(group_stats) - 5} 个群组")

            # 性能指标
            if duration > 0:
                print(f"⚡ 处理速度: {len(result) / duration:.2f} 记忆/秒")

        elif debug:
            print("ℹ️  未发现需要处理的数据（可能时间范围内没有新的MemCell）")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        if debug:
            import traceback

            traceback.print_exc()
        return False


def main():
    """主函数"""
    # 解析命令行参数
    days = 7
    debug = False

    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
        except ValueError:
            print("❌ 天数参数必须是数字")
            return 1

    if len(sys.argv) > 2 and sys.argv[2].lower() in ['debug', 'verbose', 'd', 'v']:
        debug = True

    print("🧪 memorize_offline 离线记忆处理测试")
    print(f"📋 参数: {days}天, 调试模式={'开启' if debug else '关闭'}")
    print("=" * 50)

    try:
        result = asyncio.run(test_memorize_offline(days, debug))
        if result:
            print("\n🎉 测试成功!")
            return 0
        else:
            print("\n💥 测试失败!")
            return 1
    except KeyboardInterrupt:
        print("\n⏹️  用户中断测试")
        return 2
    except Exception as e:
        print(f"\n💥 执行异常: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
