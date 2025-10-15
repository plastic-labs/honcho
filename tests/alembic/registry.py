"""Registry for migration-specific prepare and verification hooks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy.engine import Connection

if TYPE_CHECKING:
    from tests.alembic.verifier import MigrationVerifier


BeforeUpgradeHook = Callable[[Connection], None]
AfterUpgradeHook = Callable[["MigrationVerifier"], None]


@dataclass(slots=True)
class RevisionHooks:
    """Container for lifecycle hooks tied to a revision."""

    before_upgrade: list[BeforeUpgradeHook] = field(default_factory=list)
    after_upgrade: list[AfterUpgradeHook] = field(default_factory=list)


_REGISTRY: dict[str, RevisionHooks] = {}


def register_before_upgrade(
    revision: str,
) -> Callable[[BeforeUpgradeHook], BeforeUpgradeHook]:
    """Register a callable to execute before upgrading to ``revision``."""

    def decorator(func: BeforeUpgradeHook) -> BeforeUpgradeHook:
        hooks = _REGISTRY.setdefault(revision, RevisionHooks())
        hooks.before_upgrade.append(func)
        return func

    return decorator


def register_after_upgrade(
    revision: str,
) -> Callable[[AfterUpgradeHook], AfterUpgradeHook]:
    """Register a callable to execute after upgrading to ``revision``."""

    def decorator(func: AfterUpgradeHook) -> AfterUpgradeHook:
        hooks = _REGISTRY.setdefault(revision, RevisionHooks())
        hooks.after_upgrade.append(func)
        return func

    return decorator


def get_registered_hooks() -> Mapping[str, RevisionHooks]:
    """Return a snapshot of the registered revision hooks."""

    return dict(_REGISTRY)


__all__ = [
    "RevisionHooks",
    "register_before_upgrade",
    "register_after_upgrade",
    "get_registered_hooks",
]
