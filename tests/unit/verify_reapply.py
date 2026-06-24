"""Quick verification that migration re-apply worked."""
import os
from sqlalchemy import create_engine, inspect, text

engine = create_engine(os.environ.get("DB_CONNECTION_URI", "postgresql+psycopg://postgres:postgres@database:5432/honcho"))
inspector = inspect(engine)
tables = set(inspector.get_table_names())
expected = {"edges", "access_log", "context_index", "thread_binding_registry"}
present = tables & expected
print(f"Tables after re-apply: {present}")
if present == expected:
    print("✅ All tables present after re-apply")
else:
    print(f"❌ Missing: {expected - present}")

with engine.connect() as conn:
    r = conn.execute(text("SELECT version_num FROM alembic_version"))
    print(f"Alembic version: {r.scalar()}")
