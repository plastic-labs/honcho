"""Unit tests for the cache client's _redact_cache_url helper."""

import pytest

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

    # --- Secrets in query parameters ---
    # redis-py accepts ?password= (querystring options become client
    # kwargs) and cashews accepts ?secret= (HMAC signing key), so both
    # are real configuration paths that must not reach the logs.

    @pytest.mark.parametrize("param", ["password", "secret", "PASSWORD"])
    def test_query_param_secret_masked(self, param: str):
        result = _redact_cache_url(f"redis://host:6379/0?{param}=s3cret")
        assert "s3cret" not in result
        assert f"{param}=***" in result

    def test_query_param_masking_preserves_other_params(self):
        result = _redact_cache_url("redis://host:6379/0?db=1&password=s3cret&ssl=true")
        assert "s3cret" not in result
        assert "db=1" in result
        assert "ssl=true" in result

    def test_userinfo_and_query_secret_both_masked(self):
        result = _redact_cache_url("redis://:hunter2@host:6379/0?secret=s3cret")
        assert "hunter2" not in result
        assert "s3cret" not in result

    def test_non_secret_query_params_unchanged(self):
        url = "redis://localhost:6379/0?suppress=true"
        assert _redact_cache_url(url) == url

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

    def test_invalid_port_redacts_password(self):
        """Regression test for two review findings: accessing
        ``parsed.port`` on a URL with a non-numeric port raises
        ``ValueError`` (must not crash startup inside an except block),
        and the fallback must never echo the raw URL back — the
        password has to be masked even when the port is unparseable.
        """
        result = _redact_cache_url("redis://:pass@host:notaport/0")
        assert "pass" not in result
        assert "***" in result

    def test_out_of_range_port_redacts_password(self):
        result = _redact_cache_url("redis://:supersecret@host:99999/0")
        assert "supersecret" not in result
        assert "***" in result

    def test_unparseable_url_never_echoed(self):
        # Unbalanced IPv6 bracket makes urlparse itself raise; the
        # fallback must return a placeholder, not the raw input.
        result = _redact_cache_url("redis://:secret@[::1:6379/0")
        assert "secret" not in result

    def test_missing_scheme_never_echoed(self):
        # Without "redis://" urlparse sees no netloc, so the userinfo
        # (and its password) is invisible to .password — the string
        # must not be echoed back.
        result = _redact_cache_url(":hunter2@host:6379/0")
        assert "hunter2" not in result

    def test_garbage_input_does_not_raise(self):
        assert isinstance(_redact_cache_url("not a url at all"), str)

    def test_empty_string_does_not_raise(self):
        assert isinstance(_redact_cache_url(""), str)
