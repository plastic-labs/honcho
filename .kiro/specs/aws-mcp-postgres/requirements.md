# Requirements Document

## Introduction

This feature adds support for connecting Honcho to an AWS RDS PostgreSQL instance using AWS IAM authentication, and introduces an MCP (Model Context Protocol) server tool that provisions and manages AWS credentials for this connection. The goal is to enable secure, token-based database authentication via AWS IAM instead of static username/password credentials, and to expose credential management through the existing MCP server architecture.

## Glossary

- **Honcho_API**: The FastAPI-based backend application serving the Honcho REST API.
- **MCP_Server**: The Model Context Protocol server (TypeScript/Cloudflare Worker) that exposes tools for interacting with Honcho.
- **RDS_Instance**: An AWS Relational Database Service PostgreSQL instance with IAM authentication enabled.
- **IAM_Auth_Token**: A short-lived authentication token generated via the AWS RDS IAM authentication mechanism, used in place of a static database password.
- **AWS_Credential_Provider**: A module responsible for obtaining and refreshing AWS credentials (access key, secret key, session token) from the runtime environment (environment variables, instance profile, ECS task role, or explicit configuration).
- **DB_Engine**: The SQLAlchemy async engine that manages the PostgreSQL connection pool.
- **Connection_URI**: The PostgreSQL connection string used by SQLAlchemy to connect to the database.
- **Config_System**: The Pydantic-based settings system (`AppSettings`, `DBSettings`) that loads configuration from environment variables, `.env` files, and `config.toml`.

## Requirements

### Requirement 1: AWS IAM Authentication for RDS PostgreSQL

**User Story:** As a platform operator, I want Honcho to authenticate to an AWS RDS PostgreSQL instance using IAM authentication tokens, so that I can eliminate static database passwords and leverage AWS IAM policies for access control.

#### Acceptance Criteria

1. WHEN `DB_AUTH_METHOD` is set to `iam`, THE AWS_Credential_Provider SHALL generate an IAM_Auth_Token using the AWS RDS `generate-db-auth-token` API with the configured region, hostname, port, and database username.
2. WHEN the DB_Engine creates a new database connection and `DB_AUTH_METHOD` is `iam`, THE DB_Engine SHALL use a freshly generated IAM_Auth_Token as the password in the connection parameters.
3. THE IAM_Auth_Token SHALL have a maximum lifetime of 15 minutes as enforced by AWS, and THE AWS_Credential_Provider SHALL generate a new token for each new connection request.
4. WHILE `DB_AUTH_METHOD` is set to `password` (the default), THE DB_Engine SHALL use the static credentials from `CONNECTION_URI` as it does today, with no behavioral change.
5. IF the AWS_Credential_Provider fails to generate an IAM_Auth_Token (due to missing credentials, network error, or insufficient IAM permissions), THEN THE Honcho_API SHALL log the error with a descriptive message and raise a connection error.
6. WHEN `DB_AUTH_METHOD` is set to `iam`, THE DB_Engine SHALL require SSL for the database connection, as AWS RDS IAM authentication mandates encrypted connections.

### Requirement 2: AWS Database Configuration Settings

**User Story:** As a platform operator, I want to configure AWS RDS connection parameters through the existing configuration system, so that I can manage deployment settings consistently across environments.

#### Acceptance Criteria

1. THE Config_System SHALL support a `DB_AUTH_METHOD` setting with allowed values `password` and `iam`, defaulting to `password`.
2. WHEN `DB_AUTH_METHOD` is `iam`, THE Config_System SHALL require the following settings: `DB_AWS_REGION`, `DB_RDS_HOSTNAME`, `DB_RDS_PORT`, and `DB_RDS_USERNAME`.
3. THE Config_System SHALL support an optional `DB_AWS_PROFILE` setting for specifying a named AWS credentials profile.
4. THE Config_System SHALL load AWS-related database settings from environment variables, `.env` files, and `config.toml` following the existing precedence order (environment variables > `.env` > `config.toml` > defaults).
5. IF `DB_AUTH_METHOD` is `iam` and any required AWS setting (`DB_AWS_REGION`, `DB_RDS_HOSTNAME`, `DB_RDS_PORT`, `DB_RDS_USERNAME`) is missing, THEN THE Config_System SHALL raise a validation error at startup with a message identifying the missing setting.
6. WHEN `DB_AUTH_METHOD` is `iam`, THE Config_System SHALL accept an optional `DB_RDS_SSL_CA_BUNDLE` setting pointing to the path of the AWS RDS CA certificate bundle for SSL verification.

### Requirement 3: MCP Server AWS Credential Tool

**User Story:** As a developer using the MCP server, I want a tool that reports the status of AWS credentials and RDS connectivity, so that I can diagnose connection issues from within my AI-assisted workflow.

#### Acceptance Criteria

1. THE MCP_Server SHALL expose an `aws_rds_status` tool that returns the current database authentication method (`password` or `iam`), the RDS hostname, port, region, and whether the connection is healthy.
2. WHEN the `aws_rds_status` tool is invoked, THE MCP_Server SHALL call the Honcho_API health endpoint and report the database connectivity status.
3. IF the Honcho_API health check fails, THEN THE `aws_rds_status` tool SHALL return an error result with a descriptive message including the failure reason.
4. THE MCP_Server SHALL register the `aws_rds_status` tool alongside existing tool registrations (workspace, peers, sessions, conclusions, system).

### Requirement 4: Connection Pool Compatibility with IAM Tokens

**User Story:** As a platform operator, I want the connection pool to work correctly with short-lived IAM tokens, so that connections are always authenticated with valid credentials.

#### Acceptance Criteria

1. WHEN `DB_AUTH_METHOD` is `iam`, THE DB_Engine SHALL use a SQLAlchemy event listener on the `do_connect` event to inject a fresh IAM_Auth_Token into connection parameters before each new physical connection is established.
2. WHEN `DB_AUTH_METHOD` is `iam`, THE DB_Engine SHALL enable `pool_pre_ping` to detect and discard stale connections whose IAM tokens have expired.
3. WHILE `DB_AUTH_METHOD` is `iam`, THE DB_Engine SHALL set `pool_recycle` to a value no greater than 900 seconds (15 minutes) to ensure connections are recycled before IAM token expiry.
4. THE DB_Engine SHALL maintain the existing connection pool configuration options (pool size, max overflow, timeout) regardless of the authentication method.

### Requirement 5: Docker and Deployment Support for AWS IAM Authentication

**User Story:** As a DevOps engineer, I want the Docker deployment to support AWS IAM authentication for RDS, so that I can deploy Honcho on AWS infrastructure (ECS, EKS) with IAM role-based database access.

#### Acceptance Criteria

1. THE Dockerfile SHALL include the AWS RDS CA certificate bundle so that SSL connections to RDS can be verified.
2. THE docker-compose.yml.example SHALL include a commented-out example configuration section demonstrating AWS RDS IAM authentication settings.
3. THE .env.template SHALL include documented entries for all AWS RDS IAM authentication settings (`DB_AUTH_METHOD`, `DB_AWS_REGION`, `DB_RDS_HOSTNAME`, `DB_RDS_PORT`, `DB_RDS_USERNAME`, `DB_AWS_PROFILE`, `DB_RDS_SSL_CA_BUNDLE`).
4. WHEN deploying on AWS ECS or EKS, THE AWS_Credential_Provider SHALL automatically discover IAM credentials from the task role or pod service account without requiring explicit access key configuration.

### Requirement 6: Database Migration Compatibility

**User Story:** As a platform operator, I want Alembic database migrations to work with AWS IAM authentication, so that schema changes can be applied to the RDS instance using the same authentication method.

#### Acceptance Criteria

1. WHEN `DB_AUTH_METHOD` is `iam`, THE `init_db` function SHALL generate a fresh IAM_Auth_Token and construct a connection URI for Alembic to use during migrations.
2. THE Alembic migration runner SHALL use the same SSL configuration as the main application when `DB_AUTH_METHOD` is `iam`.
3. IF the IAM_Auth_Token generation fails during migration, THEN THE `init_db` function SHALL log the error and terminate with a non-zero exit code.
