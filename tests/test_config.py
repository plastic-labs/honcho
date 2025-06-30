from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic_settings import SettingsConfigDict

from src.config import (
    AppSettings,
    AuthSettings,
    HonchoSettings,
    TomlConfigSettingsSource,
    load_toml_config,
)


def test_load_toml_config_file_not_found():
    """Test that load_toml_config returns an empty dict if file not found."""
    assert load_toml_config("non_existent_file.toml") == {}


def test_load_toml_config_toml_decode_error(tmp_path: Path, caplog):
    """Test that load_toml_config handles TOML decode errors gracefully."""
    # Create a file with invalid TOML content
    invalid_toml_file = tmp_path / "invalid.toml"
    invalid_toml_file.write_text("invalid toml content [unclosed section")

    result = load_toml_config(str(invalid_toml_file))

    # Should return empty dict on decode error
    assert result == {}
    # Should log the warning
    assert "Failed to load" in caplog.text
    assert str(invalid_toml_file) in caplog.text


def test_load_toml_config_os_error(tmp_path: Path, caplog):
    """Test that load_toml_config handles OS errors gracefully."""
    # Create a file that exists but will cause an OS error when opened
    config_file = tmp_path / "config.toml"
    config_file.write_text("[section]\nkey = 'value'")

    # Mock the open function to raise an OSError
    with patch("builtins.open", side_effect=OSError("Permission denied")):
        result = load_toml_config(str(config_file))

    # Should return empty dict on OS error
    assert result == {}
    # Should log the warning
    assert "Failed to load" in caplog.text
    assert str(config_file) in caplog.text


def test_toml_config_source_with_unmapped_prefix(monkeypatch, tmp_path: Path):
    """
    Test that TomlConfigSettingsSource can handle prefixes not in SECTION_MAP
    by converting them to lowercase for the section name. This covers lines 69-70.
    """
    # 1. Create a dummy config.toml
    config_content = """
[unmapped]
test_field = "hello from toml"
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)

    # 2. Patch TOML_CONFIG to use our dummy config
    monkeypatch.setattr("src.config.TOML_CONFIG", load_toml_config(str(config_path)))

    # 3. Define a settings class with an unmapped prefix
    class UnmappedSettings(HonchoSettings):
        model_config = SettingsConfigDict(env_prefix="UNMAPPED_")
        TEST_FIELD: str = "default"

    # 4. Instantiate the settings class
    settings = UnmappedSettings()

    # 5. Assert the value is loaded from the TOML file
    assert settings.TEST_FIELD == "hello from toml"


def test_toml_config_source_prefix_cleanup(monkeypatch, tmp_path: Path):
    """
    Test that TomlConfigSettingsSource correctly strips trailing underscore from prefix.
    This covers lines 64-66 where prefix is cleaned up.
    """
    # 1. Create a dummy config.toml
    config_content = """
[testprefix]
test_field = "value_from_toml"
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)

    # 2. Patch TOML_CONFIG to use our dummy config
    monkeypatch.setattr("src.config.TOML_CONFIG", load_toml_config(str(config_path)))

    # 3. Define a settings class with a prefix ending in underscore
    class TestPrefixSettings(HonchoSettings):
        model_config = SettingsConfigDict(
            env_prefix="TESTPREFIX_"
        )  # Note the trailing underscore
        TEST_FIELD: str = "default"

    # 4. Instantiate the settings class
    settings = TestPrefixSettings()

    # 5. Assert the value is loaded from the TOML file
    # This proves that "TESTPREFIX_" was stripped to "TESTPREFIX" and then lowercased to "testprefix"
    assert settings.TEST_FIELD == "value_from_toml"


def test_toml_config_source_prefix_no_underscore(monkeypatch, tmp_path: Path):
    """
    Test that TomlConfigSettingsSource handles prefixes without trailing underscore.
    This covers lines 64-66 by testing the case where prefix doesn't end with underscore.
    """
    # 1. Create a dummy config.toml
    config_content = """
[nounder]
test_field = "value_without_underscore"
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)

    # 2. Patch TOML_CONFIG to use our dummy config
    monkeypatch.setattr("src.config.TOML_CONFIG", load_toml_config(str(config_path)))

    # 3. Define a settings class with a prefix NOT ending in underscore
    class NoUnderscoreSettings(HonchoSettings):
        model_config = SettingsConfigDict(
            env_prefix="NOUNDER"
        )  # No trailing underscore
        TEST_FIELD: str = "default"

    # 4. Instantiate the settings class
    settings = NoUnderscoreSettings()

    # 5. Assert the value is loaded from the TOML file
    # This tests line 64 (get call) and lines 65-66 (conditional that's false)
    assert settings.TEST_FIELD == "value_without_underscore"


def test_toml_config_source_get_field_value_direct(monkeypatch, tmp_path: Path):
    """
    Test TomlConfigSettingsSource.get_field_value directly to ensure lines 64-66 are covered.
    """
    from pydantic.fields import FieldInfo

    # 1. Create a dummy config.toml
    config_content = """
[testnounder]
field_name = "direct_test_value"
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)

    # 2. Patch TOML_CONFIG to use our dummy config
    monkeypatch.setattr("src.config.TOML_CONFIG", load_toml_config(str(config_path)))

    # 3. Create settings classes to test both cases
    class SettingsWithUnderscore(HonchoSettings):
        model_config = SettingsConfigDict(env_prefix="TESTNOUNDER_")

    class SettingsWithoutUnderscore(HonchoSettings):
        model_config = SettingsConfigDict(env_prefix="TESTNOUNDER")

    # 4. Test prefix with underscore (covers lines 64-66, true branch)
    source_with_underscore = TomlConfigSettingsSource(SettingsWithUnderscore)
    field_info = FieldInfo()
    result = source_with_underscore.get_field_value(field_info, "field_name")
    assert result[0] == "direct_test_value"  # Value found

    # 5. Test prefix without underscore (covers lines 64-66, false branch)
    source_without_underscore = TomlConfigSettingsSource(SettingsWithoutUnderscore)
    result = source_without_underscore.get_field_value(field_info, "field_name")
    assert result[0] == "direct_test_value"  # Value found


def test_toml_config_source_field_case_variations(monkeypatch, tmp_path: Path):
    """
    Test TomlConfigSettingsSource field name case variations.
    This covers lines 75 and 77 in get_field_value method.
    """
    from pydantic.fields import FieldInfo

    # 1. Create a config.toml with uppercase and exact case field names
    config_content = """
[testcase]
UPPER_FIELD = "uppercase_value"
ExactCase = "exact_case_value"
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)

    # 2. Patch TOML_CONFIG to use our dummy config
    monkeypatch.setattr("src.config.TOML_CONFIG", load_toml_config(str(config_path)))

    # 3. Create a settings class
    class TestCaseSettings(HonchoSettings):
        model_config = SettingsConfigDict(env_prefix="TESTCASE_")

    # 4. Test finding field by uppercase name (line 75)
    source = TomlConfigSettingsSource(TestCaseSettings)
    field_info = FieldInfo()

    # Field name is lowercase but should find UPPER_FIELD in TOML
    result = source.get_field_value(field_info, "upper_field")
    assert result[0] == "uppercase_value"  # Should find via uppercase variation

    # 5. Test finding field by exact case (line 77)
    # Field name matches exactly
    result = source.get_field_value(field_info, "ExactCase")
    assert result[0] == "exact_case_value"  # Should find via exact case variation


def test_app_settings_invalid_log_level():
    """Test that AppSettings raises ValueError for invalid log level (line 271)."""
    with pytest.raises(ValueError, match="Invalid log level: INVALID"):
        AppSettings(LOG_LEVEL="INVALID")


def test_toml_config_source_case_fallback_to_exact_match(monkeypatch, tmp_path: Path):
    """
    Test TomlConfigSettingsSource field name case fallback logic.
    This covers lines 73-77: when lowercase and uppercase don't match,
    it falls back to exact case match.
    """
    from pydantic.fields import FieldInfo

    # Create a config.toml with field name that only matches exact case
    config_content = """
[testfallback]
MixedCase_Field = "exact_match_value"
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)

    # Patch TOML_CONFIG to use our dummy config
    monkeypatch.setattr("src.config.TOML_CONFIG", load_toml_config(str(config_path)))

    # Create a settings class
    class TestFallbackSettings(HonchoSettings):
        model_config = SettingsConfigDict(env_prefix="TESTFALLBACK_")

    # Test finding field by exact case when lower/upper don't match
    source = TomlConfigSettingsSource(TestFallbackSettings)
    field_info = FieldInfo()

    # This should:
    # Line 73: field_name.lower() = "mixedcase_field" -> None (no match)
    # Line 75: field_name.upper() = "MIXEDCASE_FIELD" -> None (no match)
    # Line 77: field_name = "MixedCase_Field" -> "exact_match_value" (match!)
    result = source.get_field_value(field_info, "MixedCase_Field")
    assert result[0] == "exact_match_value"


def test_toml_config_source_get_field_value_return_format(monkeypatch, tmp_path: Path):
    """
    Test that TomlConfigSettingsSource.get_field_value returns the correct tuple format.
    This specifically tests line 79 to ensure the return statement is executed.
    """
    from pydantic.fields import FieldInfo

    # Create a config.toml with a test field
    config_content = """
[testreturn]
test_field = "test_value"
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)

    # Patch TOML_CONFIG to use our dummy config
    monkeypatch.setattr("src.config.TOML_CONFIG", load_toml_config(str(config_path)))

    # Create a settings class
    class TestReturnSettings(HonchoSettings):
        model_config = SettingsConfigDict(env_prefix="TESTRETURN_")

    # Test that get_field_value returns the correct tuple format
    source = TomlConfigSettingsSource(TestReturnSettings)
    field_info = FieldInfo()

    # This should execute line 79: return field_value, field_name, False
    result = source.get_field_value(field_info, "test_field")

    # Verify the return format is a tuple with 3 elements
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "test_value"  # field_value
    assert result[1] == "test_field"  # field_name
    assert result[2] is False  # always False as per line 79


def test_toml_config_source_get_field_value_none_return(monkeypatch, tmp_path: Path):
    """
    Test that TomlConfigSettingsSource.get_field_value returns None when field not found.
    This ensures line 79 is covered when field_value is None.
    """
    from pydantic.fields import FieldInfo

    # Create a config.toml without the field we're looking for
    config_content = """
[testreturn]
other_field = "other_value"
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)

    # Patch TOML_CONFIG to use our dummy config
    monkeypatch.setattr("src.config.TOML_CONFIG", load_toml_config(str(config_path)))

    # Create a settings class
    class TestReturnSettings(HonchoSettings):
        model_config = SettingsConfigDict(env_prefix="TESTRETURN_")

    # Test that get_field_value returns None when field not found
    source = TomlConfigSettingsSource(TestReturnSettings)
    field_info = FieldInfo()

    # This should execute line 79: return field_value, field_name, False
    # where field_value is None because "nonexistent_field" doesn't exist
    result = source.get_field_value(field_info, "nonexistent_field")

    # Verify the return format is a tuple with 3 elements and field_value is None
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] is None  # field_value should be None
    assert result[1] == "nonexistent_field"  # field_name
    assert result[2] is False  # always False as per line 79


def test_auth_settings_jwt_secret_required():
    """Test that AuthSettings raises ValueError if USE_AUTH is true and JWT_SECRET is not set."""
    with pytest.raises(ValueError, match="JWT_SECRET must be set if USE_AUTH is true"):
        AuthSettings(USE_AUTH=True, JWT_SECRET=None)
