"""
Exception classes for CrewAI integration.
"""


class HonchoDependencyError(ImportError):
    """Raised when required CrewAI dependencies are not installed."""

    def __init__(self, framework: str, install_command: str) -> None:
        self.framework = framework
        self.install_command = install_command
        super().__init__(
            f"{framework} dependencies not found. Install with: {install_command}"
        )
