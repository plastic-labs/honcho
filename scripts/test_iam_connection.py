"""Quick smoke test for IAM RDS authentication.

Usage:
    DB_AUTH_METHOD=iam \
    DB_AWS_REGION=us-east-1 \
    DB_RDS_HOSTNAME=your-host.rds.amazonaws.com \
    DB_RDS_PORT=5432 \
    DB_RDS_USERNAME=iam_db_user \
    DB_CONNECTION_URI=postgresql+psycopg://iam_db_user@your-host:5432/postgres \
    uv run python scripts/test_iam_connection.py
"""

import asyncio
import sys


async def main():
    # Import after env vars are set so settings pick them up
    from src.config import settings
    from src.db import engine

    print(f"Auth method: {settings.DB.AUTH_METHOD}")
    print(f"RDS hostname: {settings.DB.RDS_HOSTNAME}")
    print(f"RDS port: {settings.DB.RDS_PORT}")
    print(f"RDS username: {settings.DB.RDS_USERNAME}")
    print(f"AWS region: {settings.DB.AWS_REGION}")
    print()

    try:
        from sqlalchemy import text

        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT current_user, version()"))
            row = result.fetchone()
            print(f"Connected as: {row[0]}")
            print(f"PostgreSQL: {row[1]}")
            print()
            print("IAM authentication is working!")
            return 0
    except Exception as e:
        print(f"Connection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
