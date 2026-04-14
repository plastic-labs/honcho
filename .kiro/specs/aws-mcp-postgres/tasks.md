# Tasks: AWS MCP Postgres

## Task 1: Add AWS IAM auth fields to DBSettings
- [x] 1.1 Add `AUTH_METHOD`, `AWS_REGION`, `RDS_HOSTNAME`, `RDS_PORT`, `RDS_USERNAME`, `AWS_PROFILE`, and `RDS_SSL_CA_BUNDLE` fields to `DBSettings` in `src/config.py`
- [x] 1.2 Add `model_validator` to `DBSettings` that validates required IAM fields (`AWS_REGION`, `RDS_HOSTNAME`, `RDS_PORT`, `RDS_USERNAME`) are set when `AUTH_METHOD=iam`, with error messages naming the missing field(s)
- [x] 1.3 Add `[db]` section entries for new fields in `config.toml.example`
- [x] 1.4 Add documented entries for `DB_AUTH_METHOD`, `DB_AWS_REGION`, `DB_RDS_HOSTNAME`, `DB_RDS_PORT`, `DB_RDS_USERNAME`, `DB_AWS_PROFILE`, `DB_RDS_SSL_CA_BUNDLE` in `.env.template`

## Task 2: Implement AWS credential provider module
- [x] 2.1 Create `src/aws_auth.py` with `generate_rds_auth_token(region, hostname, port, username, profile)` function using boto3
- [x] 2.2 Add `boto3` dependency to `pyproject.toml`
- [x] 2.3 Implement descriptive error wrapping for `NoCredentialsError`, `ClientError`, and `EndpointConnectionError`

## Task 3: Update database engine for IAM authentication
- [x] 3.1 Update `src/db.py` to construct base connection URI from RDS settings when `AUTH_METHOD=iam` (no password in URI)
- [x] 3.2 Register `do_connect` event listener on the engine that calls `generate_rds_auth_token` and injects the token as password in `cparams`
- [x] 3.3 Force `pool_pre_ping=True` and clamp `pool_recycle` to `min(configured, 900)` when `AUTH_METHOD=iam`
- [x] 3.4 Add SSL connect args (`sslmode=require`, optional `sslrootcert` from `RDS_SSL_CA_BUNDLE`) when `AUTH_METHOD=iam`
- [x] 3.5 Ensure `password` mode behavior is completely unchanged (no event listener, no SSL override)

## Task 4: Update migration support for IAM authentication
- [x] 4.1 Update `init_db()` in `src/db.py` to generate a fresh IAM token and construct a connection URI for Alembic when `AUTH_METHOD=iam`
- [x] 4.2 Pass SSL configuration to Alembic connection when `AUTH_METHOD=iam`
- [x] 4.3 Ensure token generation failure during migration logs error and raises (non-zero exit)

## Task 5: Add MCP aws_rds_status tool
- [x] 5.1 Create `mcp/src/tools/aws-status.ts` with `register` function that registers the `aws_rds_status` tool
- [x] 5.2 Implement tool to call Honcho API `/health` endpoint and return `auth_method`, `rds_hostname`, `rds_port`, `aws_region`, `connection_healthy` fields
- [x] 5.3 Implement error handling: return `errorResult` with failure reason when health check fails
- [x] 5.4 Register the new tool in `mcp/src/server.ts` alongside existing tool registrations

## Task 6: Update Docker and deployment files
- [x] 6.1 Add `RUN` step in `Dockerfile` to download AWS RDS CA certificate bundle (`global-bundle.pem`)
- [x] 6.2 Add commented-out AWS RDS IAM authentication section to `docker-compose.yml.example`

## Task 7: Write unit tests
- [x] 7.1 Write unit tests for `DBSettings` validation: valid password config, valid iam config, missing required IAM fields, invalid auth_method
- [x] 7.2 Write unit tests for `generate_rds_auth_token`: successful token generation (mocked boto3), error cases (NoCredentialsError, ClientError)
- [x] 7.3 Write unit tests for engine creation: password mode (no listener, standard URI), iam mode (listener registered, SSL args)
- [x] 7.4 Write unit tests for `init_db` IAM path: correct URI construction, token failure handling
- [x] 7.5 Write unit tests for MCP `aws_rds_status` tool: healthy response, failed response

## Task 8: Write property-based tests
- [x] 8.1 [PBT] Property 1: Token generation uses configured parameters — generate random (region, hostname, port, username) → verify boto3 called with exact values
- [x] 8.2 [PBT] Property 2: Fresh token injection on every connection — generate random connection event sequences → verify each gets unique token
- [x] 8.3 [PBT] Property 3: Password mode preserves existing behavior — generate random CONNECTION_URI strings → verify passthrough unchanged
- [x] 8.4 [PBT] Property 4: Token generation errors are descriptive — generate random boto3 exceptions → verify descriptive error messages
- [x] 8.5 [PBT] Property 5: IAM mode enforces SSL — generate random IAM configs → verify sslmode=require and optional sslrootcert
- [x] 8.6 [PBT] Property 6: Auth method validation rejects invalid values — generate random non-password/non-iam strings → verify rejection
- [x] 8.7 [PBT] Property 7: IAM mode requires AWS fields with descriptive errors — generate random subsets of missing fields → verify error names them
- [x] 8.8 [PBT] Property 8: MCP status tool returns all required fields — generate random health responses → verify all fields present
- [x] 8.9 [PBT] Property 9: MCP status tool error includes failure reason — generate random error responses → verify reason included
- [x] 8.10 [PBT] Property 10: IAM mode forces pool_pre_ping — generate random POOL_PRE_PING values → verify always True for iam
- [x] 8.11 [PBT] Property 11: IAM mode clamps pool_recycle — generate random POOL_RECYCLE values → verify min(value, 900)
- [x] 8.12 [PBT] Property 12: Pool settings preserved across auth methods — generate random pool configs → verify preservation
- [x] 8.13 [PBT] Property 13: Migration constructs IAM URI with fresh token — generate random IAM configs → verify URI components
- [x] 8.14 [PBT] Property 14: Migration uses SSL config for IAM — generate random IAM configs → verify Alembic SSL matches engine
- [x] 8.15 [PBT] Property 15: Migration token failure terminates with error — generate random exceptions → verify propagation
