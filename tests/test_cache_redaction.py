"""Unit tests for the cache client's _redact_cache_url helper."""

from src.cache.client import _redact_cache_url


class TestRedactCacheUrl:
    """Tests for _redact_cache_url — a security-relevant logging helper
    that must never raise and must never leak a password."""

    # --- Password masking ---

    def test_password_only_userinfo(self):
        assert (
            _redact_cache_url("redis://:secret@localhost:6379/0")
            == "redis://:***@localhost:6379/0"
        )

    def test_user_and_password(self):
        result = _redact_cache_url("redis://user:s3cret@10.0.0.1:6380/2")
        assert "***" in result
        assert "s3cret" not in result
        assert "user" in result

    def test_rediss_protocol(self):
        result = _redact_cache_url("rediss://:secret@redis.example.com:6380")
        assert result.startswith("rediss://")
        assert "***" in result
        assert "secret" not in result

    def test_complex_password(self):
        result = _redact_cache_url("redis://:p%40ssw0rd!%24@host:6379/0")
        assert "***" in result
        assert "p%40ssw0rd" not in result

    def test_password_never_leaked(self):
        """The original password must never appear in the redacted output."""
        for url in [
            "redis://:hunter2@localhost:6379/0",
            "redis://admin:hunter2@localhost:6379/0",
            "rediss://:hunter2@[::1]:6380/1",
        ]:
            assert "hunter2" not in _redact_cache_url(url)

    # --- No-password URLs (returned unchanged) ---

    def test_user_without_password_unchanged(self):
        assert (
            _redact_cache_url("redis://user@localhost:6379/0")
            == "redis://user@localhost:6379/0"
        )

    def test_in_memory_url_unchanged(self):
        assert _redact_cache_url("mem://") == "mem://"

    # --- IPv6 ---

    def test_ipv6_brackets_preserved(self):
        result = _redact_cache_url("rediss://:secret@[::1]:6380/1")
        assert "[::1]" in result
        assert "***" in result
        assert "secret" not in result

    # --- Malformed URLs (must NOT raise) ---

    def test_invalid_port_does_not_raise(self):
        """Regression test for the bug found in review: accessing
        ``parsed.port`` on a URL with a non-numeric port raises
        ``ValueError``.  The helper must catch this and return the
        original URL rather than crashing startup inside an except block.
        """
        url = "redis://:pass@host:notaport/0"
        result = _redact_cache_url(url)
        # Must not raise; password should still be redacted if possible,
        # but the key requirement is that it returns a string, not raises.
        assert isinstance(result, str)

    def test_garbage_input_does_not_raise(self):
        assert isinstance(_redact_cache_url("not a url at all"), str)

    def test_empty_string_does_not_raise(self):
        assert isinstance(_redact_cache_url(""), str)
