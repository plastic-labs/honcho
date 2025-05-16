"""add_collection_id_partition_to_documents_table

Revision ID: a5c5965af1a6
Revises: 20f89a421aff
Create Date: 2025-05-16 11:05:03.907208

"""
from os import getenv
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'a5c5965af1a6'
down_revision: Union[str, None] = '20f89a421aff'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    schema = getenv("DATABASE_SCHEMA", "public")

    # 1. Create documents_temp table with minimal constraints
    op.execute(f"""
    CREATE TABLE {schema}.documents_temp (
        id BIGINT GENERATED ALWAYS AS IDENTITY,
        collection_id TEXT NOT NULL,
        public_id TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{{}}',
        content TEXT NOT NULL,
        embedding vector(1536),
        created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
        user_id TEXT NOT NULL,
        app_id TEXT NOT NULL
    ) PARTITION BY HASH (collection_id);
    """)

    # 2. Add constraints
    op.execute(f"""
    ALTER TABLE {schema}.documents_temp 
        ADD CONSTRAINT pk_documents_temp PRIMARY KEY (collection_id, id),
        ADD CONSTRAINT fk_documents_temp_collection_id_collections FOREIGN KEY (collection_id) 
            REFERENCES {schema}.collections(public_id),
        ADD CONSTRAINT fk_documents_temp_user_id_users FOREIGN KEY (user_id)
            REFERENCES {schema}.users(public_id),
        ADD CONSTRAINT fk_documents_temp_app_id_apps FOREIGN KEY (app_id)
            REFERENCES {schema}.apps(public_id),
        ADD CONSTRAINT public_id_length CHECK (length(public_id) = 21),
        ADD CONSTRAINT content_length CHECK (length(content) <= 65535),
        ADD CONSTRAINT public_id_format CHECK (public_id ~ '^[A-Za-z0-9_-]+$');
    """)

    # 3. Create indexes
    op.execute(f"""
    CREATE UNIQUE INDEX uq_documents_temp_collection_public_id ON {schema}.documents_temp (collection_id, public_id);
    CREATE INDEX ix_documents_temp_created_at ON {schema}.documents_temp (created_at);
    CREATE INDEX ix_documents_temp_user_id ON {schema}.documents_temp (user_id);
    CREATE INDEX ix_documents_temp_app_id ON {schema}.documents_temp (app_id);
    """)

    # 4. Create partitions
    op.execute(
        """
        DO $$
        BEGIN
            FOR i IN 0..7 LOOP
                EXECUTE format(
                    'CREATE TABLE documents_temp_%s PARTITION OF documents_temp 
                     FOR VALUES WITH (MODULUS 8, REMAINDER %s)', 
                    i, i
                );
            END LOOP;
        END $$;
        """
    )

    # 5. Copy data from documents to documents_temp in batches
    op.execute(f"""
    DO $$
    DECLARE
        batch_size INT := 1000;
        last_id INT := 0;
        total_copied INT := 0;
        batch_count INT;
    BEGIN
        LOOP
            -- Insert next batch
            WITH batch AS (
                SELECT 
                    collection_id, public_id, metadata, content, 
                    embedding, created_at, user_id, app_id
                FROM {schema}.documents
                WHERE id > last_id
                ORDER BY id
                LIMIT batch_size
                FOR UPDATE SKIP LOCKED
            )
            INSERT INTO {schema}.documents_temp (
                collection_id, public_id, metadata, content,
                embedding, created_at, user_id, app_id
            )
            SELECT 
                collection_id, public_id, metadata, content,
                embedding, created_at, user_id, app_id
            FROM batch
            RETURNING id INTO last_id;

            GET DIAGNOSTICS batch_count = ROW_COUNT;
            total_copied := total_copied + batch_count;
            
            RAISE NOTICE 'Copied % rows. Total: %', batch_count, total_copied;

            EXIT WHEN batch_count < batch_size;
            COMMIT;
        END LOOP;
    END $$;
    """)

    # 6. Rename the documents table to documents_old
    op.execute(f"""
    DO $$
    DECLARE
        idx_record record;
        const_record record;
    BEGIN
        -- Rename all indexes
        FOR idx_record IN 
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = '{schema}' 
              AND tablename = 'documents'
              AND indexname NOT LIKE '%_old'
        LOOP
            EXECUTE format('ALTER INDEX IF EXISTS {schema}.%I RENAME TO %I', 
                         idx_record.indexname, 
                         idx_record.indexname || '_old');
        END LOOP;

        -- Rename all constraints
        FOR const_record IN 
            SELECT conname 
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid 
            JOIN pg_namespace n ON t.relnamespace = n.oid
            WHERE n.nspname = '{schema}'
              AND t.relname = 'documents'
              AND conname NOT LIKE '%_old'
        LOOP
            EXECUTE format('ALTER TABLE IF EXISTS {schema}.documents RENAME CONSTRAINT %I TO %I', 
                         const_record.conname, 
                         const_record.conname || '_old');
        END LOOP;
    END $$;

    -- Now rename the table
    ALTER TABLE {schema}.documents RENAME TO documents_old;
    """)

    # 7. Rename the documents_temp table to documents
    op.execute(f"""
    ALTER TABLE {schema}.documents_temp RENAME TO documents;
    """)

    # 8. Rename the partitions to match the new table name
    op.execute(f"""
    DO $$
    BEGIN
        FOR i IN 0..7 LOOP
            EXECUTE format(
                'ALTER TABLE documents_temp_%s RENAME TO documents_%s',
                i, i
            );
        END LOOP;
    END $$;
    """)

    # 9. Rename constraints and indexes to match the new table name
    op.execute(f"""
    -- Rename Primary Key constraint
    ALTER TABLE {schema}.documents RENAME CONSTRAINT pk_documents_temp TO pk_documents;
    
    -- Rename Foreign Key constraints
    ALTER TABLE {schema}.documents RENAME CONSTRAINT fk_documents_temp_collection_id_collections TO fk_documents_collection_id_collections;
    ALTER TABLE {schema}.documents RENAME CONSTRAINT fk_documents_temp_user_id_users TO fk_documents_user_id_users;
    ALTER TABLE {schema}.documents RENAME CONSTRAINT fk_documents_temp_app_id_apps TO fk_documents_app_id_apps;
    
    -- Rename Indexes
    ALTER INDEX {schema}.uq_documents_temp_collection_public_id RENAME TO uq_documents_collection_public_id;
    ALTER INDEX {schema}.ix_documents_temp_created_at RENAME TO ix_documents_created_at;
    ALTER INDEX {schema}.ix_documents_temp_user_id RENAME TO ix_documents_user_id;
    ALTER INDEX {schema}.ix_documents_temp_app_id RENAME TO ix_documents_app_id;
    """)

def downgrade() -> None:
    schema = getenv("DATABASE_SCHEMA", "public")
    
    # 1. First rename the current documents table to documents_new
    op.execute(f"""
    DO $$
    DECLARE
        idx_record record;
        const_record record;
    BEGIN
        -- Drop new constraints first to avoid conflicts
        FOR const_record IN 
            SELECT conname 
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid 
            JOIN pg_namespace n ON t.relnamespace = n.oid
            WHERE n.nspname = '{schema}'
              AND t.relname = 'documents'
              AND conname NOT LIKE '%_old'
        LOOP
            EXECUTE format('ALTER TABLE IF EXISTS {schema}.documents DROP CONSTRAINT IF EXISTS %I', 
                         const_record.conname);
        END LOOP;

        -- Drop new indexes to avoid conflicts
        FOR idx_record IN 
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = '{schema}' 
              AND tablename = 'documents'
              AND indexname NOT LIKE '%_old'
        LOOP
            EXECUTE format('DROP INDEX IF EXISTS {schema}.%I', idx_record.indexname);
        END LOOP;
    END $$;

    ALTER TABLE {schema}.documents RENAME TO documents_new;
    """)

    # 2. Rename documents_old back to documents and restore original names
    op.execute(f"""
    DO $$
    DECLARE
        idx_record record;
        const_record record;
    BEGIN
        -- First rename the table back
        ALTER TABLE {schema}.documents_old RENAME TO documents;

        -- Restore original index names by removing _old suffix
        FOR idx_record IN 
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = '{schema}' 
              AND tablename = 'documents'
              AND indexname LIKE '%_old'
        LOOP
            EXECUTE format('ALTER INDEX IF EXISTS {schema}.%I RENAME TO %I', 
                         idx_record.indexname,
                         substring(idx_record.indexname from 1 for length(idx_record.indexname) - 4));
        END LOOP;

        -- Restore original constraint names by removing _old suffix
        FOR const_record IN 
            SELECT conname 
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid 
            JOIN pg_namespace n ON t.relnamespace = n.oid
            WHERE n.nspname = '{schema}'
              AND t.relname = 'documents'
              AND conname LIKE '%_old'
        LOOP
            EXECUTE format('ALTER TABLE IF EXISTS {schema}.documents RENAME CONSTRAINT %I TO %I', 
                         const_record.conname,
                         substring(const_record.conname from 1 for length(const_record.conname) - 4));
        END LOOP;
    END $$;
    """)

    # 3. Rename the partitions to documents_new_*
    op.execute(f"""
    DO $$
    BEGIN
        FOR i IN 0..7 LOOP
            EXECUTE format(
                'ALTER TABLE {schema}.documents_%s RENAME TO documents_new_%s',
                i, i
            );
        END LOOP;
    END $$;
    """)


    # 4. Drop the partitioned table
    op.execute(f"DROP TABLE IF EXISTS {schema}.documents_new CASCADE;")

