"""
Exception classes for Honcho Agno integration.
"""


class HonchoDependencyError(ImportError):
    """Raised when required Agno dependencies are not installed."""

    def __init__(self, framework: str, install_command: str) -> None:
        self.framework = framework
        self.install_command = install_command
        super().__init__(
            f"{framework} dependencies not found. Install with: {install_command}"
        )


class HonchoSessionError(Exception):
    """Raised when there is an error with Honcho session operations."""

    pass


class HonchoToolError(Exception):
    """Raised when a Honcho tool operation fails."""

    pass
