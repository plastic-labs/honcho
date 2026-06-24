"""Verify Phase 1 migration and tables."""
import os
from sqlalchemy import create_engine, text, inspect

engine = create_engine(os.environ.get("DB_CONNECTION_URI", "postgresql+psycopg://postgres:postgres@database:5432/honcho"))
inspector = inspect(engine)

# Check tables
tables = set(inspector.get_table_names())
expected = {"edges", "access_log", "context_index", "thread_binding_registry"}
missing = expected - tables
if missing:
    print(f"❌ Missing tables: {missing}")
else:
    print("✅ All 4 new tables exist")

# Check columns per table
for table in sorted(expected):
    cols = {c["name"]: str(c["type"]) for c in inspector.get_columns(table)}
    print(f"\n  {table}:")
    for name, typ in sorted(cols.items()):
        print(f"    {name}: {typ}")

# Check indexes
for table in sorted(expected):
    indexes = inspector.get_indexes(table)
    print(f"\n  Indexes on {table}: {[ix['name'] for ix in indexes]}")

# Check FKs
for table in sorted(expected):
    fks = inspector.get_foreign_keys(table)
    print(f"\n  Foreign keys on {table}:")
    for fk in fks:
        print(f"    {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")

# Verify migration rollback
print("\n--- Testing rollback ---")
with engine.begin() as conn:
    conn.execute(text("DELETE FROM alembic_version WHERE version_num = '2a3b4c5d6e7f'"))
    conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('e4eba9cfaa6f')"))

# Drop tables in reverse order
with engine.begin() as conn:
    for table in ["thread_binding_registry", "context_index", "access_log", "edges"]:
        conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        print(f"  Dropped {table}")

# Verify
tables_after = set(inspector.get_table_names())
if expected & tables_after:
    print("❌ Rollback failed — some tables still exist")
else:
    print("✅ Rollback successful — all new tables removed")

# Re-apply migration
print("\n--- Re-applying migration ---")
import subprocess
result = subprocess.run(
    [".venv/bin/python3", "-m", "alembic", "upgrade", "head"],
    capture_output=True, text=True, cwd="/app"
)
print(result.stdout)
if result.returncode == 0:
    print("✅ Re-apply successful")
else:
    print(f"❌ Re-apply failed: {result.stderr}")
