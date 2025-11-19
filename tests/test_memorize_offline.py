#!/usr/bin/env python3
"""
memorize_offline 离线记忆处理测试脚本

使用方法:
    python tests/test_memorize_offline.py                                    # 默认测试最近7天
    python tests/test_memorize_offline.py 3                                  # 测试最近3天
    python tests/test_memorize_offline.py 1 debug                           # 测试最近1天，详细输出
    python tests/test_memorize_offline.py --from 2025-09-18 --to 2025-09-19 # 自定义时间范围
    python tests/test_memorize_offline.py --from "2025-09-18 10:00" --to "2025-09-19 18:00" debug # 自定义时间+调试
    python tests/test_memorize_offline.py --extract-part personal_profile      # 仅提取个人档案
"""

import asyncio
import sys
import argparse
from datetime import datetime, timedelta
from typing import Optional

# 添加src目录到Python路径

from memory_layer.memory_manager import MemorizeOfflineRequest
from biz_layer.tanka_memorize import memorize_offline
from common_utils.datetime_utils import get_now_with_timezone
from core.observation.logger import get_logger

# 获取日志记录器
logger = get_logger(__name__)


def parse_datetime(date_str: str) -> datetime:
    """
    解析时间字符串为datetime对象

    支持格式:
    - 2025-09-18
    - 2025-09-18 10:00
    - 2025-09-18 10:00:30
    """
    try:
        # 尝试不同的时间格式
        formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                # 转换为带时区的datetime
                return get_now_with_timezone().replace(
                    year=dt.year,
                    month=dt.month,
                    day=dt.day,
                    hour=dt.hour,
                    minute=dt.minute,
                    second=dt.second,
                    microsecond=0,
                )
            except ValueError:
                continue

        raise ValueError(f"无法解析时间格式: {date_str}")

    except Exception as e:
        raise ValueError(f"时间解析失败: {e}")


async def test_memorize_offline(
    days=None,
    start_time=None,
    end_time=None,
    debug: bool = False,
    extract_part: Optional[str] = None,
):
    """测试memorize_offline流程"""

    # 确定时间范围
    if start_time and end_time:
        # 使用自定义时间范围
        memorize_from = start_time
        memorize_to = end_time
        logger.info("🚀 测试memorize_offline流程（自定义时间范围）")
    elif days:
        # 使用天数范围
        current_time = get_now_with_timezone()
        memorize_from = current_time - timedelta(days=days)
        memorize_to = current_time
        logger.info(f"🚀 测试memorize_offline流程（最近{days}天）")
    else:
        # 默认7天
        current_time = get_now_with_timezone()
        memorize_from = current_time - timedelta(days=7)
        memorize_to = current_time
        logger.info("🚀 测试memorize_offline流程（默认最近7天）")

    normalized_extract_part = None
    if extract_part:
        normalized = extract_part.strip().lower()
        if normalized and normalized != "all":
            normalized_extract_part = normalized

    request = MemorizeOfflineRequest(
        memorize_from=memorize_from,
        memorize_to=memorize_to,
        extract_part=normalized_extract_part,
    )

    logger.info(
        f"⏰ 时间范围: {request.memorize_from.strftime('%Y-%m-%d %H:%M')} ~ {request.memorize_to.strftime('%Y-%m-%d %H:%M')}"
    )
    if normalized_extract_part:
        logger.info(f"🎯 提取范围: {normalized_extract_part}")
    else:
        logger.info("🎯 提取范围: all")

    try:
        # 执行测试
        start_time = datetime.now()
        result = await memorize_offline(request)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        logger.info(f"✅ 测试完成! 耗时: {duration:.2f}秒")
        logger.info(f"📊 提取记忆: {len(result) if result else 0} 个")

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

            logger.info("📈 记忆类型分布:")
            for memory_type, count in type_stats.items():
                logger.info(f"   {memory_type}: {count} 个")

            if debug and group_stats:
                logger.debug("👥 群组分布:")
                for group_id, count in list(group_stats.items())[:5]:  # 只显示前5个
                    logger.debug(f"   {group_id}: {count} 个")
                if len(group_stats) > 5:
                    logger.debug(f"   ... 还有 {len(group_stats) - 5} 个群组")

            # 性能指标
            if duration > 0:
                logger.info(f"⚡ 处理速度: {len(result) / duration:.2f} 记忆/秒")

        elif debug:
            logger.debug("ℹ️  未发现需要处理的数据（可能时间范围内没有新的MemCell）")

        return True

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}", exc_info=debug)
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='memorize_offline 离线记忆处理测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s                                    # 默认测试最近7天
  %(prog)s 3                                  # 测试最近3天
  %(prog)s 1 debug                           # 测试最近1天，详细输出
  %(prog)s --from 2025-09-18 --to 2025-09-19 # 自定义时间范围
  %(prog)s --from "2025-09-18 10:00" --to "2025-09-19 18:00" debug # 自定义时间+调试
  %(prog)s --extract-part personal_profile    # 仅提取个人档案
        """,
    )

    # 位置参数（兼容旧用法）
    parser.add_argument('days', nargs='?', type=int, help='测试天数（从现在往前推N天）')
    parser.add_argument('mode', nargs='?', help='模式：debug/verbose/d/v 开启调试输出')

    # 自定义时间范围
    parser.add_argument(
        '--from',
        dest='start_time',
        help='开始时间 (格式: 2025-09-18 或 "2025-09-18 10:00")',
    )
    parser.add_argument(
        '--to',
        dest='end_time',
        help='结束时间 (格式: 2025-09-19 或 "2025-09-19 18:00")',
    )

    # 调试模式
    parser.add_argument('--debug', '-d', action='store_true', help='开启调试模式')
    parser.add_argument(
        '--verbose', '-v', action='store_true', help='开启详细输出（同--debug）'
    )
    parser.add_argument(
        '--extract-part',
        choices=['all', 'personal_profile', 'group_profile'],
        help='指定提取范围，默认all',
    )

    args = parser.parse_args()

    # 确定调试模式
    debug = (
        args.debug
        or args.verbose
        or (args.mode and args.mode.lower() in ['debug', 'verbose', 'd', 'v'])
    )

    # 验证参数
    days = args.days
    start_time = None
    end_time = None

    if args.start_time and args.end_time:
        # 自定义时间范围模式
        try:
            start_time = parse_datetime(args.start_time)
            end_time = parse_datetime(args.end_time)

            if start_time >= end_time:
                logger.error("❌ 开始时间必须早于结束时间")
                return 1

        except ValueError as e:
            logger.error(f"❌ 时间格式错误: {e}")
            logger.error(
                "支持的格式: 2025-09-18 或 '2025-09-18 10:00' 或 '2025-09-18 10:00:30'"
            )
            return 1

    elif args.start_time or args.end_time:
        logger.error("❌ 必须同时指定 --from 和 --to 参数")
        return 1

    # 如果没有指定天数且没有自定义时间，使用默认值
    if not days and not (start_time and end_time):
        days = 7

    extract_part = args.extract_part
    extract_part_display = extract_part or 'all'
    if extract_part_display == 'all':
        extract_part_arg = None
    else:
        extract_part_arg = extract_part_display

    logger.info("🧪 memorize_offline 离线记忆处理测试")
    if start_time and end_time:
        logger.info(
            f"📋 参数: 自定义时间范围, 调试模式={'开启' if debug else '关闭'}, 提取范围={extract_part_display}"
        )
    else:
        logger.info(
            f"📋 参数: {days}天, 调试模式={'开启' if debug else '关闭'}, 提取范围={extract_part_display}"
        )
    logger.info("=" * 50)

    try:
        result = asyncio.run(
            test_memorize_offline(
                days=days,
                start_time=start_time,
                end_time=end_time,
                debug=debug,
                extract_part=extract_part_arg,
            )
        )
        if result:
            logger.info("\n🎉 测试成功!")
            return 0
        else:
            logger.error("\n💥 测试失败!")
            return 1
    except KeyboardInterrupt:
        logger.warning("\n⏹️  用户中断测试")
        return 2
    except Exception as e:
        logger.error(f"\n💥 执行异常: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
