"""
Test None handling logic of datetime_utils.to_iso_format function
"""

import datetime
import pytest
from zoneinfo import ZoneInfo

from common_utils.datetime_utils import to_iso_format, get_timezone


class TestToIsoFormatNoneHandling:
    """Test to_iso_format function's handling of None values"""

    def test_none_with_allow_none_true_returns_none(self):
        """When allow_none=True (default) and None is passed, should return None"""
        result = to_iso_format(None)
        assert result is None

    def test_none_with_allow_none_true_explicit_returns_none(self):
        """When allow_none=True is explicitly set and None is passed, should return None"""
        result = to_iso_format(None, allow_none=True)
        assert result is None

    def test_none_with_allow_none_false_raises_value_error(self):
        """When allow_none=False and None is passed, should raise ValueError"""
        with pytest.raises(ValueError) as exc_info:
            to_iso_format(None, allow_none=False)

        # Verify error message
        assert "time_value cannot be None" in str(exc_info.value)
        assert "allow_none=False" in str(exc_info.value)


class TestToIsoFormatNormalCases:
    """Test to_iso_format function with normal input cases (ensure changes don't break existing functionality)"""

    def test_datetime_input(self):
        """Test datetime object input"""
        tz = get_timezone()
        dt = datetime.datetime(2025, 12, 5, 10, 30, 0, tzinfo=tz)
        result = to_iso_format(dt)

        assert result is not None
        assert "2025-12-05" in result
        assert "10:30:00" in result

    def test_datetime_input_with_allow_none_false(self):
        """Test datetime object input with allow_none=False (should not affect normal input)"""
        tz = get_timezone()
        dt = datetime.datetime(2025, 12, 5, 10, 30, 0, tzinfo=tz)
        result = to_iso_format(dt, allow_none=False)

        assert result is not None
        assert "2025-12-05" in result

    def test_timestamp_seconds_input(self):
        """Test timestamp input in seconds"""
        # Timestamp for 2024-12-05 10:30:00 UTC
        timestamp = 1733394600
        result = to_iso_format(timestamp)

        assert result is not None
        assert "2024-12-05" in result

    def test_timestamp_milliseconds_input(self):
        """Test timestamp input in milliseconds"""
        # Millisecond timestamp for 2024-12-05 10:30:00 UTC
        timestamp_ms = 1733394600000
        result = to_iso_format(timestamp_ms)

        assert result is not None
        assert "2024-12-05" in result

    def test_string_input_passthrough(self):
        """Test string input returns directly"""
        iso_str = "2025-12-05T10:30:00+00:00"
        result = to_iso_format(iso_str)

        assert result == iso_str

    def test_string_input_with_allow_none_false(self):
        """Test string input with allow_none=False"""
        iso_str = "2025-12-05T10:30:00+00:00"
        result = to_iso_format(iso_str, allow_none=False)

        assert result == iso_str

    def test_empty_string_returns_none(self):
        """Test empty string returns None"""
        result = to_iso_format("")
        assert result is None

    def test_negative_timestamp_returns_none(self):
        """Test negative timestamp returns None"""
        result = to_iso_format(-1)
        assert result is None

    def test_zero_timestamp_returns_none(self):
        """Test zero timestamp returns None"""
        result = to_iso_format(0)
        assert result is None


class TestToIsoFormatEdgeCases:
    """测试 to_iso_format 函数的边界情况"""

    def test_float_timestamp(self):
        """测试浮点数时间戳"""
        # 2024-12-05 10:30:00.123 UTC 的浮点数时间戳
        timestamp = 1733394600.123
        result = to_iso_format(timestamp)

        assert result is not None
        assert "2024-12-05" in result

    def test_datetime_without_timezone(self):
        """测试不带时区的 datetime 对象（应自动添加时区）"""
        dt = datetime.datetime(2025, 12, 5, 10, 30, 0)
        result = to_iso_format(dt)

        assert result is not None
        # 应该包含时区信息
        assert "+" in result or "-" in result

    def test_unsupported_type_returns_none(self):
        """测试不支持的类型返回 None"""
        result = to_iso_format([1, 2, 3])  # type: ignore
        assert result is None
