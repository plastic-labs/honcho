These tests validate Alembic migrations end to end for structure, order, and correctness.
They ensure reversibility, expected schema and data, and integration with the registry and pipeline.
The key components are the verifier, the test pipeline, the registry, and the revisions under test.

### Verifier

- The verifier runs checks when specific revisions are applied and reverted.
- It validates the schema after upgrade, verifies data migrations such as defaults, backfills, and transforms, and confirms reversibility after downgrade.
- Assertions are grouped per revision or feature, and helpers use the SQLAlchemy inspector to introspect the database.

### Test Pipeline

- The pipeline orchestrates the database lifecycle by creating a database, applying upgrades and downgrades, running verifications, and tearing down resources.
- It typically starts from base, upgrades to the revision immediately before a target revision, seeds the DB, runs the target migration, and then validates the schema + data
- It relies on shared fixtures such as `engine`, `connection`, and `alembic_config`, and it ensures isolation per test

### Registry

- The registry declares revisions and test metadata used to drive scenarios.
- It defines ordering and selection, attaches verifier callbacks to specific revisions or ranges, and centralizes per revision expectations.

### Revisions

- Eevisions are the migration scripts under `alembic/revisions`
- Each revision should provide functions decorated with `register_before_upgrade()` and `register_after_upgrade()`. These are used to validate schemas and data before and after a migration is run

### Running the Tests

- Tests can be run all together `pytest tests/alembic` or individually `pytest tests/alembic -k "revision_number"`. For example, to run the test against a1b2c3d4e5f6_initial_schema.py, you would run the command `pytest tests/alembic -k "a1b2c3d4e5f6"`
