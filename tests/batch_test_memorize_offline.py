#!/usr/bin/env python3
"""
批量运行 memorize_offline 测试脚本

这个脚本会将指定的日期范围按月拆分，然后逐月调用 test_memorize_offline.py 中的测试函数。
直接导入函数调用，支持断点调试。

使用方法:
    python tests/batch_test_memorize_offline.py --from 2025-09-01 --to 2025-12-01
    python tests/batch_test_memorize_offline.py --from "2025-09-01 00:00" --to "2025-12-01 00:00"
    python tests/batch_test_memorize_offline.py --from 2025-09-01 --to 2025-12-01 --debug
    python tests/batch_test_memorize_offline.py --from 2025-09-01 --to 2025-12-01 --parallel
    python tests/batch_test_memorize_offline.py --from 2025-09-01 --to 2025-12-01 --extract-part personal_profile
"""

import asyncio
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import time
import traceback
import calendar
from typing import Optional

# 添加项目路径，确保可以导入测试模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests"))

# 导入测试函数和时区工具
from test_memorize_offline import test_memorize_offline
from common_utils.datetime_utils import timezone  # type: ignore


def add_months(source_date: datetime, months: int) -> datetime:
    """
    给日期添加指定月数，保留时区信息

    Args:
        source_date: 源日期
        months: 要添加的月数

    Returns:
        添加后的日期（保留原始时区信息）
    """
    month = source_date.month - 1 + months
    year = source_date.year + month // 12
    month = month % 12 + 1
    day = min(source_date.day, calendar.monthrange(year, month)[1])

    # 使用 replace 保留时区信息
    result = source_date.replace(year=year, month=month, day=day)
    return result


def parse_datetime(date_str: str) -> datetime:
    """
    解析时间字符串为datetime对象（带时区信息）

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
                # 解析为 naive datetime 后添加时区信息
                naive_dt = datetime.strptime(date_str, fmt)
                # 使用项目统一的时区
                aware_dt = naive_dt.replace(tzinfo=timezone)
                return aware_dt
            except ValueError:
                continue

        raise ValueError(f"无法解析时间格式: {date_str}")

    except Exception as e:
        raise ValueError(f"时间解析错误: {str(e)}")


async def run_single_month_test(
    date: datetime,
    debug: bool = False,
    timeout: int = 600,
    extract_part: Optional[str] = None,
) -> tuple[str, bool, str]:
    """
    运行单月测试（异步版本，支持断点调试）

    Args:
        date: 测试月份的第一天
        debug: 是否开启调试模式
        timeout: 超时时间（秒）

    Returns:
        tuple: (月份字符串, 是否成功, 输出信息)
    """
    date_str = date.strftime('%Y-%m')
    # 月初第一天 00:00:00
    start_time_dt = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    # 下个月第一天 00:00:00
    end_time_dt = add_months(start_time_dt, 1)

    try:
        print(f"⏳ 开始处理 {date_str} (时间范围: {start_time_dt} 到 {end_time_dt})...")
        start_time = time.time()

        # 直接调用测试函数（支持断点调试）
        success = await asyncio.wait_for(
            test_memorize_offline(
                start_time=start_time_dt,
                end_time=end_time_dt,
                debug=debug,
                extract_part=extract_part,
            ),
            timeout=timeout,
        )

        end_time = time.time()
        elapsed = end_time - start_time

        if success:
            print(f"✅ {date_str} 处理完成 (耗时: {elapsed:.2f}秒)")
            output = f"✅ {date_str}: 成功 (耗时: {elapsed:.2f}秒)"
        else:
            print(f"❌ {date_str} 处理失败 (耗时: {elapsed:.2f}秒)")
            output = f"❌ {date_str}: 失败 (耗时: {elapsed:.2f}秒)"

        return date_str, success, output

    except asyncio.TimeoutError:
        print(f"⏰ {date_str} 处理超时 (超过 {timeout} 秒)")
        return date_str, False, f"⏰ {date_str}: 超时 (超过 {timeout} 秒)"
    except Exception as e:
        print(f"💥 {date_str} 处理异常: {str(e)}")
        error_details = f"💥 {date_str}: 异常 - {str(e)}"
        if debug:
            error_details += f"\n{traceback.format_exc()}"
        return date_str, False, error_details


async def run_batch_tests(
    start_date: datetime,
    end_date: datetime,
    debug: bool = False,
    parallel: bool = False,
    max_workers: int = 2,
    timeout: int = 600,
    extract_part: Optional[str] = None,
):
    """
    批量运行测试（异步版本，支持断点调试）

    Args:
        start_date: 开始日期
        end_date: 结束日期
        debug: 是否开启调试模式
        parallel: 是否并行执行
        max_workers: 并行执行时的最大并发数
        timeout: 单个测试的超时时间（秒）
    """
    # 生成月份列表（每个月的第一天）
    current_date = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date_normalized = end_date.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )

    dates = []
    while current_date < end_date_normalized:
        dates.append(current_date)
        current_date = add_months(current_date, 1)

    total_months = len(dates)
    first_month_str = dates[0].strftime('%Y-%m')
    last_month_str = dates[-1].strftime('%Y-%m') if dates else first_month_str
    print(f"📅 将处理 {total_months} 个月的数据: {first_month_str} 到 {last_month_str}")
    print(f"🔧 执行模式: {'并行' if parallel else '串行'}")
    if parallel:
        print(f"🔢 最大并发数: {max_workers}")
    print(f"⏱️ 单个测试超时: {timeout} 秒")
    print(f"🎯 提取范围: {extract_part or 'all'}")
    print("=" * 60)

    results = []
    total_start_time = time.time()

    # 串行执行
    for i, date in enumerate(dates, 1):
        print(f"\n📊 进度: {i}/{total_months}")
        try:
            result = await run_single_month_test(date, debug, timeout, extract_part)
            results.append(result)
        except Exception as e:
            month_str = date.strftime('%Y-%m')
            error_msg = f"💥 {month_str}: 执行异常 - {str(e)}"
            if debug:
                error_msg += f"\n{traceback.format_exc()}"
            results.append((month_str, False, error_msg))

    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time

    # 统计结果
    successful = sum(1 for _, success, _ in results if success)
    failed = total_months - successful

    print("\n" + "=" * 60)
    print("📈 批量测试完成！")
    print(f"⏱️ 总耗时: {total_elapsed:.2f} 秒")
    print(f"✅ 成功: {successful}/{total_months}")
    print(f"❌ 失败: {failed}/{total_months}")
    print(f"📊 成功率: {successful/total_months*100:.1f}%")

    if debug or failed > 0:
        print("\n📝 详细结果:")
        # 按月份排序显示结果
        results.sort(key=lambda x: x[0])
        for month_str, success, output in results:
            print(f"{output}")

    return successful == total_months


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量运行 memorize_offline 测试脚本（按月执行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --from 2025-09-01 --to 2025-12-01                    # 串行测试3个月（支持断点调试）
  %(prog)s --from "2025-09-01 00:00" --to "2025-12-01 00:00"   # 自定义时间范围
  %(prog)s --from 2025-09-01 --to 2025-12-01 --extract-part personal_profile

注意：
  - 按月份处理数据，每个月从该月1号00:00:00到下月1号00:00:00
  - 直接导入test_memorize_offline函数，支持断点调试
  - 可在test_memorize_offline.py和相关代码中设置断点
  - 串行模式下调试体验最佳
  - 由于处理时间较长，建议单月超时至少600秒
        """,
    )

    # 必需参数
    parser.add_argument(
        '--from',
        dest='start_time',
        required=True,
        help='开始月份 (格式: 2025-09-01 或 "2025-09-01 00:00"，会自动对齐到月初)',
    )
    parser.add_argument(
        '--to',
        dest='end_time',
        required=True,
        help='结束月份 (格式: 2025-12-01 或 "2025-12-01 00:00"，会自动对齐到月初)',
    )

    # 可选参数
    parser.add_argument(
        '--debug', '-d', action='store_true', help='开启调试模式，显示详细输出'
    )
    parser.add_argument(
        '--parallel', '-p', action='store_true', help='并行执行测试（默认串行）'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=2,
        help='并行执行时的最大工作线程数（默认2，月度处理建议不超过3）',
    )
    parser.add_argument(
        '--timeout', type=int, default=600, help='单个测试的超时时间（秒，默认600）'
    )
    parser.add_argument(
        '--extract-part',
        choices=['all', 'personal_profile', 'group_profile'],
        help='指定提取范围，默认all',
    )

    args = parser.parse_args()

    # 解析时间参数
    try:
        start_time = parse_datetime(args.start_time)
        end_time = parse_datetime(args.end_time)

        # 自动对齐到月初
        start_time = start_time.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        end_time = end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        if start_time >= end_time:
            print("❌ 开始时间必须早于结束时间")
            return 1

    except ValueError as e:
        print(f"❌ 时间格式错误: {e}")
        print("支持的格式: 2025-09-01 或 '2025-09-01 00:00' 或 '2025-09-01 00:00:00'")
        return 1

    # 验证参数
    if args.max_workers < 1:
        print("❌ --max-workers 必须大于0")
        return 1

    if args.timeout < 60:
        print("❌ --timeout 必须大于等于60秒（月度处理建议至少600秒）")
        return 1

    extract_part_arg = args.extract_part
    normalized_extract_part = None
    if extract_part_arg and extract_part_arg.lower() != 'all':
        normalized_extract_part = extract_part_arg.lower()

    try:
        # 运行批量测试（异步）
        success = await run_batch_tests(
            start_time,
            end_time,
            debug=args.debug,
            parallel=args.parallel,
            max_workers=args.max_workers,
            timeout=36000,  # 10小时
            extract_part=normalized_extract_part,
        )

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断执行")
        return 1
    except Exception as e:
        print(f"💥 批量测试过程中发生异常: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
