"""drop the id-only fallback from the queue token-count function

Revision ID: 0c67be41049b
Revises: 647b773f05ac
Create Date: 2026-10-21

``_queue_item_token_count`` prices a tenant-less queue row by probing
messages under the default tenant, then, on a miss, by ``id`` alone. The
id-only probe cannot use the ``(tenant_id, id)`` primary key as a search key,
so it reads every messages partition. e5fe7f8bcf62 stamps every existing
message with the default tenant and makes the column NOT NULL, so the
fallback can only find nothing (a deleted message) or a row in another
tenant. Either way it should price the row at 0. This replaces the function
without the fallback; the triggers call it by name, so they pick it up.
"""

from collections.abc import Sequence

from alembic import op

from migrations.utils import get_schema

# revision identifiers, used by Alembic.
revision: str = "0c67be41049b"
down_revision: str | None = "647b773f05ac"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()


def _token_count_function(tenantless_probe: str) -> str:
    return f"""
        CREATE OR REPLACE FUNCTION {schema}._queue_item_token_count(
            tenant TEXT, msg_id BIGINT, item_task_type TEXT
        ) RETURNS BIGINT AS $$
        DECLARE
            t BIGINT;
        BEGIN
            IF item_task_type <> 'representation' OR msg_id IS NULL THEN
                RETURN 0;
            END IF;
            IF tenant IS NULL THEN
{tenantless_probe}
            ELSE
                SELECT token_count INTO t
                FROM {schema}.messages
                WHERE tenant_id = tenant AND id = msg_id;
            END IF;
            RETURN coalesce(t, 0);
        END $$ LANGUAGE plpgsql;
        """


def upgrade() -> None:
    op.execute(
        _token_count_function(
            f"""                SELECT token_count INTO t
                FROM {schema}.messages
                WHERE tenant_id = 'default' AND id = msg_id;"""
        )
    )


def downgrade() -> None:
    op.execute(
        _token_count_function(
            f"""                SELECT token_count INTO t
                FROM {schema}.messages
                WHERE tenant_id = 'default' AND id = msg_id;
                IF NOT FOUND THEN
                    SELECT token_count INTO t
                    FROM {schema}.messages WHERE id = msg_id;
                END IF;"""
        )
    )
