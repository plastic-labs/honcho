"""Honcho CLI — a terminal for Honcho."""

from importlib.metadata import PackageNotFoundError, version

try:
    # pyproject.toml is the single source of the version; reading it from the
    # installed metadata keeps --version, the update check and the
    # X-Honcho-Host header in step with each release bump.
    __version__ = version("honcho-cli")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0+unknown"
