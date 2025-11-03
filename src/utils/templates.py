"""
Template rendering utility for managing Jinja2 prompt templates.

This module provides a centralized template manager for rendering prompt templates
used across the application (deriver, dreamer, dialectic).
"""

import logging
from functools import lru_cache
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape
from jinja2.exceptions import TemplateNotFound, TemplateSyntaxError

logger = logging.getLogger(__name__)


class TemplateManager:
    """Manages Jinja2 template rendering for prompts."""

    env: Environment

    def __init__(self) -> None:
        """
        Initialize the template manager.

        Uses PackageLoader to load templates from the 'src' package's 'templates' directory.
        This works both when running from source and when installed via pip.
        """
        self.env = Environment(
            loader=PackageLoader("src", "templates"),
            autoescape=select_autoescape(enabled_extensions=[], default=False),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

        # Add custom filters
        def join_lines(items: Any) -> str:
            """Join items with newlines."""
            return "\n".join(items) if items else ""

        self.env.filters["join_lines"] = join_lines

    def render(self, template_name: str, context: dict[str, Any]) -> str:
        """
        Render a template with the given context.

        Args:
            template_name: Name of the template file (e.g., 'dreamer/consolidation.jinja')
            context: Dictionary of variables to pass to the template

        Returns:
            Rendered template string with leading/trailing whitespace stripped
        """
        try:
            template = self.env.get_template(template_name)
            rendered = template.render(**context)
            # Strip leading/trailing whitespace to match cleandoc behavior
            return rendered.strip()
        except TemplateNotFound as e:
            logger.exception(f"Template {template_name} not found")
            raise e
        except TemplateSyntaxError as e:
            logger.exception(f"Template syntax error in {template_name}")
            raise e
        except Exception as e:
            logger.exception(f"Error rendering template {template_name}")
            raise e


@lru_cache(maxsize=1)
def get_template_manager() -> TemplateManager:
    """
    Get a cached instance of the template manager.

    Returns:
        Singleton TemplateManager instance
    """
    return TemplateManager()


def render_template(template_name: str, context: dict[str, Any]) -> str:
    """
    Convenience function to render a template using the default template manager.

    Args:
        template_name: Name of the template file (e.g., 'dreamer/consolidation.jinja')
        context: Dictionary of variables to pass to the template

    Returns:
        Rendered template string
    """
    manager = get_template_manager()
    return manager.render(template_name, context)
