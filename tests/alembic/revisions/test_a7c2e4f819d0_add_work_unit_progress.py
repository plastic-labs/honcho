"""Migration preserves unknown start times while initializing new attempts."""

from sqlalchemy import text

from tests.alembic.registry import register_after_upgrade, register_before_upgrade
from tests.alembic.verifier import MigrationVerifier


@register_before_upgrade("a7c2e4f819d0")
def prepare_progress(verifier: MigrationVerifier) -> None:
    verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}".active_queue_sessions (id, work_unit_key) '
            + "VALUES ('progress-old', 'progress-old')"
        )
    )


@register_after_upgrade("a7c2e4f819d0")
def verify_progress(verifier: MigrationVerifier) -> None:
    for name in ("started_at", "last_activity_at", "stage", "current_queue_item_id"):
        verifier.assert_column_exists("active_queue_sessions", name, nullable=True)
    old = verifier.conn.execute(
        text(
            f'SELECT started_at, last_activity_at, stage FROM "{verifier.schema}".active_queue_sessions '
            + "WHERE id = 'progress-old'"
        )
    ).one()
    assert tuple(old) == (None, None, None)
    new = verifier.conn.execute(
        text(
            f'INSERT INTO "{verifier.schema}".active_queue_sessions (id, work_unit_key) '
            + "VALUES ('progress-new', 'progress-new') RETURNING started_at, last_activity_at, stage"
        )
    ).one()
    assert new.started_at is not None
    assert new.started_at == new.last_activity_at
    assert new.stage == "claimed"
    verifier.conn.execute(
        text(
            f'DELETE FROM "{verifier.schema}".active_queue_sessions '
            + "WHERE id IN ('progress-old', 'progress-new')"
        )
    )
