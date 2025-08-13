"""
Comprehensive tests for datetime parsing functionality in utils/formatting.py.

This test suite covers edge cases for timezone handling, format validation,
security considerations, and integration with the filtering system.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.utils.filter import (
    _validate_datetime_string,  # pyright: ignore[reportPrivateUsage]
)
from src.utils.formatting import (
    format_datetime_utc,
    parse_datetime_iso,
    utc_now_iso,
)


class TestParseDatetimeISO:
    """Test parse_datetime_iso function with various input formats and edge cases."""

    def test_parse_z_suffix_format(self):
        """Test parsing ISO strings with Z suffix (UTC indicator)."""
        test_cases = [
            "2023-01-01T12:00:00Z",
            "2023-12-31T23:59:59Z",
            "2000-02-29T00:00:00Z",  # Leap year
            "1970-01-01T00:00:00Z",  # Unix epoch
            "2023-01-01T12:00:00.123456Z",  # With microseconds
        ]

        for iso_string in test_cases:
            result = parse_datetime_iso(iso_string)
            assert result.tzinfo == timezone.utc
            assert isinstance(result, datetime)

    def test_parse_plus_zero_format(self):
        """Test parsing ISO strings with +00:00 timezone indicator."""
        test_cases = [
            "2023-01-01T12:00:00+00:00",
            "2023-12-31T23:59:59+00:00",
            "2023-01-01T12:00:00.123456+00:00",
        ]

        for iso_string in test_cases:
            result = parse_datetime_iso(iso_string)
            assert result.tzinfo == timezone.utc
            assert isinstance(result, datetime)

    def test_parse_without_timezone(self):
        """Test parsing ISO strings without timezone (assumes UTC)."""
        test_cases = [
            "2023-01-01T12:00:00",
            "2023-01-01T12:00:00.123456",
        ]

        for iso_string in test_cases:
            result = parse_datetime_iso(iso_string)
            assert result.tzinfo == timezone.utc
            assert isinstance(result, datetime)

    def test_equivalency_between_formats(self):
        """Test that Z and +00:00 formats produce equivalent results."""
        base_time = "2023-01-01T12:00:00"

        result_z = parse_datetime_iso(base_time + "Z")
        result_plus = parse_datetime_iso(base_time + "+00:00")
        result_none = parse_datetime_iso(base_time)

        assert result_z == result_plus == result_none
        assert all(
            r.tzinfo == timezone.utc for r in [result_z, result_plus, result_none]
        )

    def test_microseconds_precision(self):
        """Test handling of microseconds precision."""
        test_cases = [
            ("2023-01-01T12:00:00.000001Z", 1),  # 1 microsecond
            ("2023-01-01T12:00:00.123456Z", 123456),  # Full precision
            ("2023-01-01T12:00:00.999999Z", 999999),  # Max microseconds
        ]

        for iso_string, expected_microseconds in test_cases:
            result = parse_datetime_iso(iso_string)
            assert result.microsecond == expected_microseconds

    def test_edge_case_dates(self):
        """Test parsing of edge case dates."""
        edge_cases = [
            "1900-01-01T00:00:00Z",  # Very old date
            "2100-12-31T23:59:59Z",  # Far future date
            "2000-02-29T12:00:00Z",  # Leap year Feb 29
            "1999-02-28T12:00:00Z",  # Non-leap year Feb 28
            "2023-01-01T00:00:00Z",  # Start of day
            "2023-12-31T23:59:59Z",  # End of day
        ]

        for iso_string in edge_cases:
            result = parse_datetime_iso(iso_string)
            assert result.tzinfo == timezone.utc
            assert isinstance(result, datetime)

    def test_invalid_formats_raise_errors(self):
        """Test that invalid datetime formats raise ValueError."""
        invalid_cases = [
            "not-a-date",
            "2023-13-01T12:00:00Z",  # Invalid month
            "2023-01-32T12:00:00Z",  # Invalid day
            "2023-01-01T25:00:00Z",  # Invalid hour
            "2023-01-01T12:60:00Z",  # Invalid minute
            "2023-01-01T12:00:60Z",  # Invalid second
            "2023/01/01T12:00:00Z",  # Wrong date separator
            "",  # Empty string
        ]

        for invalid_input in invalid_cases:
            with pytest.raises(ValueError):
                parse_datetime_iso(invalid_input)

    def test_none_and_non_string_inputs(self):
        """Test that None and non-string inputs raise errors."""
        invalid_inputs: list[Any] = [None, 123, [], {}]

        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                parse_datetime_iso(invalid_input)

    def test_permissive_formats_accepted(self):
        """Test that some permissive formats are actually accepted by fromisoformat."""
        permissive_cases = [
            "2023-01-01 12:00:00",  # Space separator (valid in ISO 8601)
            "2023-01-01T12:00:00",  # No timezone (gets UTC added)
        ]

        for permissive_input in permissive_cases:
            result = parse_datetime_iso(permissive_input)
            assert isinstance(result, datetime)
            assert result.tzinfo == timezone.utc

    def test_malicious_injection_attempts(self):
        """Test that potential SQL injection attempts in datetime strings fail safely."""
        malicious_inputs = [
            "2023-01-01'; DROP TABLE messages; --",
            "2023-01-01T12:00:00Z' OR 1=1",
            "2023-01-01T12:00:00Z'; SELECT * FROM users; --",
            "2023-01-01T12:00:00Z' UNION SELECT password FROM auth",
            "2023-01-01T12:00:00Z<script>alert('xss')</script>",
            "2023-01-01T12:00:00Z\r\n",  # CRLF injection
            "2023-01-01T12:00:00Z\x00",  # Null byte
        ]

        for malicious_input in malicious_inputs:
            with pytest.raises(ValueError):
                parse_datetime_iso(malicious_input)

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        # All suspicious unicode characters should be rejected
        unicode_cases = [
            "2023-01-01T12:00:00Z\u200b",  # Zero-width space
            "2023-01-01T12:00:00Z\ufeff",  # Byte order mark
            "2023\u200001\u200001T12:00:00Z",  # Zero-width space in date
            "2023-01-01T12:00:00Z\u0000",  # Null unicode
        ]

        for unicode_input in unicode_cases:
            with pytest.raises(ValueError):
                parse_datetime_iso(unicode_input)

    def test_whitespace_handling(self):
        """Test handling of leading/trailing whitespace."""
        # Valid whitespace should be stripped and accepted
        whitespace_cases = [
            "  2023-01-01T12:00:00Z  ",  # Leading/trailing spaces
            "\t2023-01-01T12:00:00Z\t",  # Tabs
            " 2023-01-01T12:00:00Z",  # Leading space only
            "2023-01-01T12:00:00Z ",  # Trailing space only
        ]

        for whitespace_input in whitespace_cases:
            result = parse_datetime_iso(whitespace_input)
            assert isinstance(result, datetime)
            assert result.tzinfo == timezone.utc

        # Invalid whitespace with newlines should be rejected
        invalid_whitespace_cases = [
            "\n2023-01-01T12:00:00Z\n",  # Newlines
            "2023-01-01T12:00:00Z\r",  # Carriage return
            "2023-01-01T12:00:00Z\r\n",  # CRLF
        ]

        for invalid_input in invalid_whitespace_cases:
            with pytest.raises(ValueError):
                parse_datetime_iso(invalid_input)

    def test_timezone_offset_formats(self):
        """Test parsing various timezone offset formats."""
        timezone_cases = [
            ("2023-01-01T12:00:00-05:00", timedelta(hours=-5)),  # EST
            ("2023-01-01T12:00:00+05:30", timedelta(hours=5, minutes=30)),  # IST
            ("2023-01-01T12:00:00-08:00", timedelta(hours=-8)),  # PST
            ("2023-01-01T12:00:00+00:00", timedelta(0)),  # UTC explicit
        ]

        for iso_string, expected_offset in timezone_cases:
            result = parse_datetime_iso(iso_string)
            assert isinstance(result, datetime)
            assert result.tzinfo is not None
            assert result.utcoffset() == expected_offset


class TestFormatDatetimeUTC:
    """Test format_datetime_utc function for consistent output formatting."""

    def test_format_utc_datetime(self):
        """Test formatting UTC datetime objects."""
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = format_datetime_utc(dt)
        assert result == "2023-01-01T12:00:00Z"

    def test_format_naive_datetime_assumes_utc(self):
        """Test that naive datetimes are assumed to be UTC."""
        dt = datetime(2023, 1, 1, 12, 0, 0)  # No timezone
        result = format_datetime_utc(dt)
        assert result == "2023-01-01T12:00:00Z"

    def test_format_non_utc_converts_to_utc(self):
        """Test that non-UTC timezones are converted to UTC."""
        # EST is UTC-5
        est = timezone(timedelta(hours=-5))
        dt = datetime(2023, 1, 1, 7, 0, 0, tzinfo=est)  # 7 AM EST
        result = format_datetime_utc(dt)
        assert result == "2023-01-01T12:00:00Z"  # Should be 12 PM UTC

    def test_format_with_microseconds(self):
        """Test formatting datetimes with microseconds."""
        dt = datetime(2023, 1, 1, 12, 0, 0, 123456, tzinfo=timezone.utc)
        result = format_datetime_utc(dt)
        assert result == "2023-01-01T12:00:00.123456Z"

    def test_roundtrip_consistency(self):
        """Test that format -> parse -> format is consistent."""
        original_dt = datetime(2023, 1, 1, 12, 30, 45, 123456, tzinfo=timezone.utc)

        # Format to string
        formatted = format_datetime_utc(original_dt)

        # Parse back to datetime
        parsed = parse_datetime_iso(formatted)

        # Format again
        reformatted = format_datetime_utc(parsed)

        assert formatted == reformatted
        assert original_dt == parsed


class TestUTCNowISO:
    """Test utc_now_iso function for current time generation."""

    def test_returns_valid_iso_string(self):
        """Test that utc_now_iso returns a valid ISO string."""
        result = utc_now_iso()

        # Should be parseable
        parsed = parse_datetime_iso(result)
        assert isinstance(parsed, datetime)
        assert parsed.tzinfo == timezone.utc

    def test_returns_recent_time(self):
        """Test that utc_now_iso returns a recent time (within last few seconds)."""
        before = datetime.now(timezone.utc)
        result_str = utc_now_iso()
        after = datetime.now(timezone.utc)

        result = parse_datetime_iso(result_str)

        assert before <= result <= after

    def test_format_consistency(self):
        """Test that utc_now_iso uses consistent Z format."""
        result = utc_now_iso()
        assert result.endswith("Z")
        assert "+00:00" not in result


class TestFilterDatetimeValidation:
    """Test _validate_datetime_string function used in API filters."""

    def test_validate_valid_formats(self):
        """Test validation of valid datetime formats."""
        timezone_aware_cases = [
            "2023-01-01T12:00:00Z",
            "2023-01-01T12:00:00+00:00",
            "2023-01-01T12:00:00",
            "2023-01-01 12:00:00",
        ]

        for valid_input in timezone_aware_cases:
            result = _validate_datetime_string(valid_input)
            assert result is not None
            assert isinstance(result, datetime)
            assert result.tzinfo == timezone.utc  # Should have UTC timezone

        # Date-only format should also be timezone-aware
        date_only_result = _validate_datetime_string("2023-01-01")
        assert date_only_result is not None
        assert isinstance(date_only_result, datetime)
        assert date_only_result.tzinfo == timezone.utc  # Should assume UTC

    def test_validate_invalid_formats_return_none(self):
        """Test that invalid formats return None rather than raising exceptions."""
        invalid_cases = [
            "not-a-date",
            "2023-13-01",  # Invalid month
            "2023-01-32",  # Invalid day
            "",
            "2023-01-01'; DROP TABLE messages; --",  # SQL injection attempt
        ]

        for invalid_input in invalid_cases:
            result = _validate_datetime_string(invalid_input)
            assert result is None

    def test_validate_strips_whitespace(self):
        """Test that validation strips leading/trailing whitespace."""
        whitespace_input = "  2023-01-01T12:00:00Z  "
        result = _validate_datetime_string(whitespace_input)
        assert result is not None
        assert isinstance(result, datetime)

    def test_validate_security_against_injection(self):
        """Test that validation protects against injection attacks."""
        injection_attempts = [
            "2023-01-01'; DROP TABLE messages; --",
            "2023-01-01 OR 1=1",
            "'; SELECT * FROM users; --",
            "2023-01-01' UNION SELECT password FROM auth",
        ]

        for injection_attempt in injection_attempts:
            result = _validate_datetime_string(injection_attempt)
            assert result is None  # Should reject malicious input


class TestDatetimeEdgeCasesIntegration:
    """Integration tests for datetime handling across the system."""

    def test_leap_year_handling(self):
        """Test handling of leap year dates."""
        leap_year_cases = [
            "2000-02-29T12:00:00Z",  # Leap year (divisible by 400)
            "2004-02-29T12:00:00Z",  # Leap year (divisible by 4)
            "1900-02-28T12:00:00Z",  # Not leap year (divisible by 100, not 400)
        ]

        for case in leap_year_cases:
            result = parse_datetime_iso(case)
            assert isinstance(result, datetime)

            # Should be able to format back
            formatted = format_datetime_utc(result)
            assert isinstance(formatted, str)

    def test_daylight_saving_transition(self):
        """Test handling of daylight saving time transitions."""
        # Create datetime during DST transition (these are complex edge cases)
        # Spring forward: 2023-03-12 2:00 AM EST becomes 3:00 AM EDT
        # Fall back: 2023-11-05 2:00 AM EDT becomes 1:00 AM EST

        # Since our system normalizes to UTC, DST transitions shouldn't affect us
        # But we should test edge cases around the transition times
        spring_forward = "2023-03-12T07:00:00Z"  # 2 AM EST = 7 AM UTC
        fall_back = "2023-11-05T06:00:00Z"  # 1 AM EST = 6 AM UTC

        for transition_time in [spring_forward, fall_back]:
            result = parse_datetime_iso(transition_time)
            assert result.tzinfo == timezone.utc

    def test_year_boundary_handling(self):
        """Test handling of year boundary dates."""
        boundary_cases = [
            "1999-12-31T23:59:59Z",  # Last second of millennium
            "2000-01-01T00:00:00Z",  # First second of millennium
            "2023-12-31T23:59:59.999999Z",  # Last microsecond of year
            "2024-01-01T00:00:00.000000Z",  # First microsecond of year
        ]

        for boundary_case in boundary_cases:
            result = parse_datetime_iso(boundary_case)
            assert isinstance(result, datetime)
            assert result.tzinfo == timezone.utc

    def test_time_precision_limits(self):
        """Test handling of time precision edge cases."""
        precision_cases = [
            "2023-01-01T12:00:00.000000Z",  # No microseconds
            "2023-01-01T12:00:00.000001Z",  # Minimum microseconds
            "2023-01-01T12:00:00.999999Z",  # Maximum microseconds
            "2023-01-01T12:00:00.123Z",  # Partial microseconds (should pad)
        ]

        for precision_case in precision_cases:
            result = parse_datetime_iso(precision_case)
            assert isinstance(result, datetime)

            # Should be able to format and parse back consistently
            formatted = format_datetime_utc(result)
            reparsed = parse_datetime_iso(formatted)
            assert result == reparsed

    def test_extreme_date_values(self):
        """Test handling of extreme date values within reasonable bounds."""
        # Python datetime has limits: 1-01-01 to 9999-12-31
        extreme_cases = [
            "0001-01-01T00:00:00Z",  # Minimum year
            "9999-12-31T23:59:59Z",  # Maximum year
            "1970-01-01T00:00:00Z",  # Unix epoch start
            "2038-01-19T03:14:07Z",  # 32-bit timestamp limit (2^31 - 1 seconds)
        ]

        for extreme_case in extreme_cases:
            result = parse_datetime_iso(extreme_case)
            assert isinstance(result, datetime)
            assert result.tzinfo == timezone.utc

    def test_consistency_across_functions(self):
        """Test that all datetime functions work consistently together."""
        # Generate a current time
        now_str = utc_now_iso()

        # Parse it
        now_dt = parse_datetime_iso(now_str)

        # Format it back
        reformatted = format_datetime_utc(now_dt)

        # Should be parseable by filter validation
        validated = _validate_datetime_string(reformatted)

        assert validated is not None
        assert validated.tzinfo == timezone.utc
        assert abs((validated - now_dt).total_seconds()) < 1  # Should be very close


class TestTimezoneHandlingEdgeCases:
    """Test edge cases specifically related to timezone handling."""

    def test_timezone_info_preservation(self):
        """Test that timezone information is correctly handled and converted."""
        # Test various timezone inputs
        utc_dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        naive_dt = datetime(2023, 1, 1, 12, 0, 0)  # No timezone

        # Both should format to the same UTC string when naive assumes UTC
        utc_formatted = format_datetime_utc(utc_dt)
        naive_formatted = format_datetime_utc(naive_dt)

        assert utc_formatted == naive_formatted == "2023-01-01T12:00:00Z"

    def test_non_utc_timezone_conversion(self):
        """Test conversion from non-UTC timezones."""
        # Create a datetime in EST (UTC-5)
        est = timezone(timedelta(hours=-5))
        est_dt = datetime(2023, 1, 1, 7, 0, 0, tzinfo=est)

        # When formatted as UTC, should be 5 hours later
        utc_formatted = format_datetime_utc(est_dt)
        assert utc_formatted == "2023-01-01T12:00:00Z"

        # When parsed back, should be equivalent UTC time
        parsed_utc = parse_datetime_iso(utc_formatted)
        assert parsed_utc.tzinfo == timezone.utc
        assert parsed_utc.hour == 12  # 7 AM EST = 12 PM UTC

    def test_timezone_edge_cases_around_midnight(self):
        """Test timezone conversions that cross date boundaries."""
        # 1 AM EST on Jan 1 = 6 AM UTC on Jan 1
        est = timezone(timedelta(hours=-5))
        est_dt = datetime(2023, 1, 1, 1, 0, 0, tzinfo=est)
        utc_formatted = format_datetime_utc(est_dt)
        assert utc_formatted == "2023-01-01T06:00:00Z"

        # 11 PM EST on Dec 31 = 4 AM UTC on Jan 1 (crosses date boundary)
        est_dt_late = datetime(2022, 12, 31, 23, 0, 0, tzinfo=est)
        utc_formatted_late = format_datetime_utc(est_dt_late)
        assert utc_formatted_late == "2023-01-01T04:00:00Z"  # Next day in UTC


class TestErrorHandlingAndRecovery:
    """Test error handling and graceful recovery scenarios."""

    def test_partial_datetime_parsing_fallback(self):
        """Test fallback behavior for partial datetime strings."""
        # The filter validation function has fallback formats
        time_cases = [
            "2023-01-01 12:00:00",  # Space separator
            "2023-01-01T12:00:00",  # No timezone
        ]

        for partial_case in time_cases:
            result = _validate_datetime_string(partial_case)
            assert result is not None
            assert result.tzinfo == timezone.utc  # Should assume UTC

        # Date-only should also be timezone-aware with UTC assumption
        date_result = _validate_datetime_string("2023-01-01")
        assert date_result is not None
        assert date_result.tzinfo == timezone.utc  # Should assume UTC

    def test_graceful_degradation_on_errors(self):
        """Test that datetime parsing fails gracefully without crashing."""
        error_cases: list[Any] = [
            None,
            123,  # Not a string
            [],  # Wrong type
            {},  # Wrong type
        ]

        for error_case in error_cases:
            # parse_datetime_iso should raise appropriate errors
            if error_case is None or not isinstance(error_case, str):
                with pytest.raises(ValueError):
                    parse_datetime_iso(error_case)

            # Filter validation should return None for invalid types
            if isinstance(error_case, str) or error_case is None:
                result = _validate_datetime_string(error_case or "")
                assert result is None
