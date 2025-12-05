import datetime
from zoneinfo import ZoneInfo
import os
from core.observation.logger import get_logger

logger = get_logger(__name__)


def get_timezone() -> ZoneInfo:
    """
    获取时区
    """
    tz = os.getenv("TZ", "Asia/Shanghai")
    return ZoneInfo(tz)


timezone = get_timezone()


def get_now_with_timezone() -> datetime.datetime:
    """
    获取当前时间，使用本地时区
    return datetime.datetime(2025, 9, 16, 20, 17, 41, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))
    """
    return datetime.datetime.now(tz=timezone)


def to_timezone(dt: datetime.datetime, tz: ZoneInfo = None) -> datetime.datetime:
    """
    将datetime对象转换为指定时区
    """
    if tz is None:
        tz = timezone
    return dt.astimezone(tz)


def to_iso_format(dt: datetime.datetime) -> str:
    """
    将datetime对象转换为ISO格式字符串（带时区）
    return 2025-09-16T20:20:06.517301+08:00
    """
    if dt.tzinfo is None:
        # 如果没有，因为默认用的是TZ环境变量，所以需要手动设置时区
        dt = dt.replace(tzinfo=timezone)
    # 如果是utc之类的，转成本地时区
    return dt.astimezone(timezone).isoformat()


def from_timestamp(timestamp: int | float) -> datetime.datetime:
    """
    从时间戳转换为datetime对象，自动识别秒级和毫秒级精度

    Args:
        timestamp: 时间戳，支持秒级（10位数字）和毫秒级（13位数字）

    Returns:
        datetime.datetime(2025, 9, 16, 20, 17, 41, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))
    """
    # 自动识别时间戳精度
    # 毫秒级时间戳通常 >= 1e12 (1000000000000)，约13位数字
    # 秒级时间戳通常 < 1e12，约10位数字
    if timestamp >= 1e12:
        # 毫秒级时间戳，转换为秒级
        timestamp_seconds = timestamp / 1000.0
    else:
        # 秒级时间戳，直接使用
        timestamp_seconds = timestamp

    return datetime.datetime.fromtimestamp(timestamp_seconds, tz=timezone)


def to_timestamp(dt: datetime.datetime) -> int:
    """
    将datetime对象转换为时间戳，秒单位
    return 1758025061
    """
    return int(dt.timestamp())


def to_timestamp_ms(dt: datetime.datetime) -> int:
    """
    将datetime对象转换为毫秒级时间戳
    return 1758025061123
    """
    return int(dt.timestamp() * 1000)


def to_timestamp_ms_universal(time_value) -> int:
    """
    通用时间格式转毫秒级时间戳函数
    支持多种输入格式：
    - int/float: 时间戳（自动识别秒级或毫秒级）
    - str: ISO格式时间字符串
    - datetime对象
    - None: 返回0

    Args:
        time_value: 各种格式的时间值

    Returns:
        int: 毫秒级时间戳，失败时返回0
    """
    try:
        if time_value is None:
            return 0

        # 处理数字类型（时间戳）
        if isinstance(time_value, (int, float)):
            # 自动识别时间戳精度
            if time_value >= 1e12:
                # 毫秒级时间戳，直接返回
                return int(time_value)
            else:
                # 秒级时间戳，转换为毫秒级
                return int(time_value * 1000)

        # 处理字符串类型
        if isinstance(time_value, str):
            # 先尝试作为数字解析
            try:
                numeric_value = float(time_value)
                return to_timestamp_ms_universal(numeric_value)
            except ValueError:
                # 不是数字，尝试作为ISO格式时间字符串解析
                dt = from_iso_format(time_value)
                return to_timestamp_ms(dt)

        # 处理datetime对象
        if isinstance(time_value, datetime.datetime):
            return to_timestamp_ms(time_value)

        # 其他类型，尝试转换为字符串再解析
        return to_timestamp_ms_universal(str(time_value))

    except Exception as e:
        logger.error(
            "[DateTimeUtils] to_timestamp_ms_universal - Error converting time value %s: %s",
            time_value,
            str(e),
        )
        return 0


def from_datetime_str_strict(time_str: str) -> datetime.datetime:
    """
    Parse datetime string in STRICT mode - raises exception on failure.
    
    Difference from from_iso_format:
        - from_datetime_str_strict: Strict mode, raises ValueError on failure (for data import)
        - from_iso_format: Lenient mode, returns current time on failure (for runtime conversion)
    
    Supported formats (local time strings without timezone):
        - "2025-01-07 09:15:33" (space-separated, common in database exports)
        - "2025-01-07T09:15:33" (ISO T-separated)
        - "2025-01-07 09:15:33.123456" (with microseconds)
        - "2025-01-07T09:15:33.123456" (ISO with microseconds)
    
    Note: Does NOT support timezone suffixes (e.g., "Z" or "+08:00"). 
          Use from_iso_format for those formats.
    
    Args:
        time_str: Datetime string to parse
    
    Returns:
        Timezone-aware datetime object (using system TZ environment variable)
    
    Raises:
        ValueError: If time string format is invalid or cannot be parsed
    
    Example:
        >>> from_datetime_str_strict("2025-01-07 09:15:33")
        datetime.datetime(2025, 1, 7, 9, 15, 33, tzinfo=ZoneInfo('Asia/Shanghai'))
    """
    if not time_str or not isinstance(time_str, str):
        raise ValueError(f"Invalid time string: {time_str}")
    
    time_str = time_str.strip()
    
    # 尝试常见格式（按使用频率排序）
    formats = [
        "%Y-%m-%d %H:%M:%S",         # "2025-01-07 09:15:33"
        "%Y-%m-%dT%H:%M:%S",         # "2025-01-07T09:15:33"
        "%Y-%m-%d %H:%M:%S.%f",      # "2025-01-07 09:15:33.123456"
        "%Y-%m-%dT%H:%M:%S.%f",      # "2025-01-07T09:15:33.123456"
    ]
    
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(time_str, fmt)
            # 添加时区信息
            return dt.replace(tzinfo=timezone)
        except ValueError:
            continue
    
    # 所有格式都失败，抛出异常
    raise ValueError(
        f"Failed to parse datetime string '{time_str}'. "
        f"Supported formats: 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DDTHH:MM:SS' (with optional microseconds)"
    )


def from_iso_format(create_time, target_timezone: ZoneInfo = None) -> datetime.datetime:
    """
    将时间转换为带时区信息的datetime对象

    Args:
        create_time: 时间对象或字符串，如 datetime对象 或 "2025-09-15T13:11:15.588000" 或 "2025-09-15T13:11:15.588000Z"
        target_timezone: 时区对象，如果为None，则使用TZ环境变量

    Returns:
        带时区信息的datetime对象，默认为东八区时区
    """
    try:
        # 处理不同的输入类型
        if isinstance(create_time, datetime.datetime):
            # 如果已经是datetime对象，直接使用
            dt = create_time
        elif isinstance(create_time, str):
            # 兼容以 "Z" 结尾的 UTC 时间格式（如 "2025-09-15T13:11:15.588000Z"）
            # Python 3.11 之前的 fromisoformat 不支持 "Z" 后缀，需要替换为 "+00:00"
            time_str = (
                create_time.replace("Z", "+00:00")
                if create_time.endswith("Z")
                else create_time
            )
            # 如果是字符串，解析为datetime对象
            dt = datetime.datetime.fromisoformat(time_str)
        else:
            # 其他类型，尝试转换为字符串再解析
            time_str = str(create_time)
            time_str = (
                time_str.replace("Z", "+00:00") if time_str.endswith("Z") else time_str
            )
            dt = datetime.datetime.fromisoformat(time_str)

        # 如果datetime对象没有时区信息，默认为指定时区
        if dt.tzinfo is None:
            # 使用指定的时区，默认为东八区
            tz = target_timezone or get_timezone()
            dt_localized = dt.replace(tzinfo=tz)
        else:
            # 如果已有时区信息，直接使用
            dt_localized = dt

        # 统一转换为与get_timezone()一致的时区
        return dt_localized.astimezone(get_timezone())

    except Exception as e:
        # 如果转换失败，返回当前时间的东八区时区对象
        logger.error(
            "[DateTimeUtils] from_iso_format - Error converting time: %s", str(e)
        )
        return get_now_with_timezone()
