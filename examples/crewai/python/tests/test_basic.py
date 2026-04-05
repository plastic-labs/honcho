"""
Basic tests for honcho_crewai package

Validates package structure, imports, and basic functionality.
"""

import pytest


def test_package_import():
    """Test that honcho_crewai imports successfully."""
    import honcho_crewai

    assert honcho_crewai is not None


def test_storage_import():
    """Test that HonchoStorage can be imported."""
    from honcho_crewai import HonchoStorage

    assert HonchoStorage is not None


def test_tools_import():
    """Test that tool classes can be imported."""
    from honcho_crewai import (
        HonchoGetContextTool,
        HonchoDialecticTool,
        HonchoSearchTool,
    )

    assert HonchoGetContextTool is not None
    assert HonchoDialecticTool is not None
    assert HonchoSearchTool is not None


class TestPackageMetadata:
    """Test package metadata and structure."""

    def test_package_has_version(self):
        """Test that package exposes version information."""
        import honcho_crewai

        assert hasattr(honcho_crewai, "__version__")
        assert isinstance(honcho_crewai.__version__, str)
        assert len(honcho_crewai.__version__) > 0

    def test_package_all_exports(self):
        """Test that __all__ contains expected exports."""
        import honcho_crewai

        assert hasattr(honcho_crewai, "__all__")
        expected_exports = [
            "HonchoStorage",
            "HonchoGetContextTool",
            "HonchoDialecticTool",
            "HonchoSearchTool",
            "HonchoDependencyError",
        ]

        for export in expected_exports:
            assert export in honcho_crewai.__all__, f"{export} not in __all__"
