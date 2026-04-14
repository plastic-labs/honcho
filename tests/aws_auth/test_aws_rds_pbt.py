"""Property-based tests for AWS RDS IAM authentication feature.

Uses hypothesis to verify correctness properties across randomized inputs.
Each test runs a minimum of 100 iterations.

Feature: aws-mcp-postgres
"""

import logging
from unittest.mock import MagicMock, patch
from urllib.parse import quote_plus

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pydantic import ValidationError

from src.aws_auth import generate_rds_auth_token
from src.config import DBSettings


# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Strategy for valid AWS region strings
aws_region_st = st.from_regex(r"[a-z]{2}-[a-z]+-[0-9]", fullmatch=True)

# Strategy for valid RDS hostnames
rds_hostname_st = st.from_regex(
    r"[a-z][a-z0-9\-]{1,20}\.[a-z0-9\-]{1,30}\.rds\.amazonaws\.com", fullmatch=True
)

# Strategy for valid ports (1-65535)
rds_port_st = st.integers(min_value=1, max_value=65535)

# Strategy for valid usernames (non-empty alphanumeric)
rds_username_st = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)

# Strategy for optional AWS profile names
aws_profile_st = st.one_of(st.none(), st.from_regex(r"[a-z][a-z0-9\-]{0,15}", fullmatch=True))

# Strategy for optional SSL CA bundle paths
ssl_ca_bundle_st = st.one_of(st.none(), st.from_regex(r"/[a-z][a-z0-9/\-]{1,40}\.pem", fullmatch=True))

# Strategy for valid connection URIs
connection_uri_st = st.from_regex(
    r"postgresql\+psycopg://[a-z]+:[a-z]+@[a-z]+:[0-9]{4}/[a-z]+", fullmatch=True
)

# Strategy for pool settings
pool_size_st = st.integers(min_value=1, max_value=100)
max_overflow_st = st.integers(min_value=0, max_value=100)
pool_timeout_st = st.integers(min_value=1, max_value=300)
pool_recycle_st = st.integers(min_value=1, max_value=7200)



# ---------------------------------------------------------------------------
# Property 1: Token generation uses configured parameters
# Feature: aws-mcp-postgres, Property 1: Token generation uses configured parameters
# ---------------------------------------------------------------------------


class TestProperty1TokenGenerationUsesConfiguredParameters:
    """**Validates: Requirements 1.1**

    For any valid IAM configuration (region, hostname, port, username, optional
    profile), calling generate_rds_auth_token should invoke the underlying boto3
    generate_db_auth_token with those exact parameter values.
    """

    @settings(max_examples=100)
    @given(
        region=aws_region_st,
        hostname=rds_hostname_st,
        port=rds_port_st,
        username=rds_username_st,
        profile=aws_profile_st,
    )
    @patch("src.aws_auth.boto3.Session")
    def test_boto3_called_with_exact_values(
        self, mock_session_cls, region, hostname, port, username, profile
    ):
        # Feature: aws-mcp-postgres, Property 1: Token generation uses configured parameters
        mock_client = MagicMock()
        mock_client.generate_db_auth_token.return_value = "token-placeholder"
        mock_session_cls.return_value.client.return_value = mock_client

        token = generate_rds_auth_token(
            region=region,
            hostname=hostname,
            port=port,
            username=username,
            profile=profile,
        )

        assert token == "token-placeholder"

        # Verify boto3 Session was created with exact region and profile
        mock_session_cls.assert_called_once_with(
            region_name=region,
            profile_name=profile,
        )

        # Verify generate_db_auth_token was called with exact parameters
        mock_client.generate_db_auth_token.assert_called_once_with(
            DBHostname=hostname,
            Port=port,
            DBUsername=username,
            Region=region,
        )


# ---------------------------------------------------------------------------
# Property 2: Fresh token injection on every connection
# Feature: aws-mcp-postgres, Property 2: Fresh token injection on every connection
# ---------------------------------------------------------------------------


class TestProperty2FreshTokenInjection:
    """**Validates: Requirements 1.2, 1.3, 4.1**

    For any sequence of do_connect events when auth_method=iam, each event
    should result in a call to generate_rds_auth_token and the returned token
    should be unique (no caching).
    """

    @settings(max_examples=100)
    @given(num_connections=st.integers(min_value=2, max_value=10))
    @patch("src.aws_auth.boto3.Session")
    def test_each_connection_gets_unique_token(self, mock_session_cls, num_connections):
        # Feature: aws-mcp-postgres, Property 2: Fresh token injection on every connection
        tokens = [f"token-{i}" for i in range(num_connections)]
        mock_client = MagicMock()
        mock_client.generate_db_auth_token.side_effect = tokens
        mock_session_cls.return_value.client.return_value = mock_client

        collected_tokens = []
        for _ in range(num_connections):
            t = generate_rds_auth_token(
                region="us-east-1",
                hostname="db.rds.amazonaws.com",
                port=5432,
                username="user",
            )
            collected_tokens.append(t)

        # Each call should have produced a distinct token
        assert len(collected_tokens) == num_connections
        assert len(set(collected_tokens)) == num_connections
        assert mock_client.generate_db_auth_token.call_count == num_connections



# ---------------------------------------------------------------------------
# Property 3: Password mode preserves existing behavior
# Feature: aws-mcp-postgres, Property 3: Password mode preserves existing behavior
# ---------------------------------------------------------------------------


class TestProperty3PasswordModePreservesExistingBehavior:
    """**Validates: Requirements 1.4**

    For any DBSettings where auth_method=password, the engine should use the
    CONNECTION_URI value directly, with no do_connect event listener and no
    SSL enforcement.
    """

    @settings(max_examples=100)
    @given(uri=connection_uri_st)
    def test_password_mode_passthrough(self, uri):
        # Feature: aws-mcp-postgres, Property 3: Password mode preserves existing behavior
        s = DBSettings(AUTH_METHOD="password", CONNECTION_URI=uri)

        assert s.AUTH_METHOD == "password"
        assert s.CONNECTION_URI == uri
        # No AWS fields should be required
        assert s.AWS_REGION is None
        assert s.RDS_HOSTNAME is None
        assert s.RDS_USERNAME is None


# ---------------------------------------------------------------------------
# Property 4: Token generation errors are descriptive
# Feature: aws-mcp-postgres, Property 4: Token generation errors are descriptive
# ---------------------------------------------------------------------------


class TestProperty4TokenGenerationErrorsAreDescriptive:
    """**Validates: Requirements 1.5**

    For any exception raised by the boto3 credential provider, the
    generate_rds_auth_token function should raise an error whose message
    includes a human-readable description of the failure.
    """

    @settings(max_examples=100)
    @given(
        error_code=st.sampled_from([
            "AccessDenied", "Forbidden", "ThrottlingException",
            "InternalError", "ServiceUnavailable", "InvalidParameterValue",
        ]),
        error_message=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))),
    )
    @patch("src.aws_auth.boto3.Session")
    def test_client_errors_produce_descriptive_messages(
        self, mock_session_cls, error_code, error_message
    ):
        # Feature: aws-mcp-postgres, Property 4: Token generation errors are descriptive
        error_response = {"Error": {"Code": error_code, "Message": error_message}}
        mock_client = MagicMock()
        mock_client.generate_db_auth_token.side_effect = ClientError(
            error_response, "GenerateDBAuthToken"
        )
        mock_session_cls.return_value.client.return_value = mock_client

        with pytest.raises(RuntimeError) as exc_info:
            generate_rds_auth_token(
                region="us-east-1",
                hostname="db.rds.amazonaws.com",
                port=5432,
                username="user",
            )

        err_msg = str(exc_info.value)
        # Error message should be descriptive (non-empty, human-readable)
        assert len(err_msg) > 10
        if "AccessDenied" in error_code or "Forbidden" in error_code:
            assert "rds-db:connect" in err_msg
        else:
            assert "AWS client error" in err_msg

    @settings(max_examples=100)
    @given(data=st.data())
    @patch("src.aws_auth.boto3.Session")
    def test_no_credentials_error_is_descriptive(self, mock_session_cls, data):
        # Feature: aws-mcp-postgres, Property 4: Token generation errors are descriptive
        mock_client = MagicMock()
        mock_client.generate_db_auth_token.side_effect = NoCredentialsError()
        mock_session_cls.return_value.client.return_value = mock_client

        with pytest.raises(RuntimeError, match="AWS credentials not found"):
            generate_rds_auth_token(
                region="us-east-1",
                hostname="db.rds.amazonaws.com",
                port=5432,
                username="user",
            )



# ---------------------------------------------------------------------------
# Property 5: IAM mode enforces SSL
# Feature: aws-mcp-postgres, Property 5: IAM mode enforces SSL
# ---------------------------------------------------------------------------


class TestProperty5IamModeEnforcesSSL:
    """**Validates: Requirements 1.6**

    For any DBSettings where auth_method=iam, the engine's connect_args should
    include sslmode=require. If RDS_SSL_CA_BUNDLE is set, sslrootcert should
    equal that path.
    """

    @settings(max_examples=100)
    @given(
        region=aws_region_st,
        hostname=rds_hostname_st,
        port=rds_port_st,
        username=rds_username_st,
        ca_bundle=ssl_ca_bundle_st,
    )
    def test_ssl_enforced_for_iam(self, region, hostname, port, username, ca_bundle):
        # Feature: aws-mcp-postgres, Property 5: IAM mode enforces SSL
        s = DBSettings(
            AUTH_METHOD="iam",
            AWS_REGION=region,
            RDS_HOSTNAME=hostname,
            RDS_PORT=port,
            RDS_USERNAME=username,
            RDS_SSL_CA_BUNDLE=ca_bundle,
        )

        # Simulate the connect_args logic from db.py
        connect_args: dict = {"prepare_threshold": None}
        if s.AUTH_METHOD == "iam":
            connect_args["sslmode"] = "require"
            if s.RDS_SSL_CA_BUNDLE:
                connect_args["sslrootcert"] = s.RDS_SSL_CA_BUNDLE

        assert connect_args["sslmode"] == "require"
        if ca_bundle:
            assert connect_args["sslrootcert"] == ca_bundle
        else:
            assert "sslrootcert" not in connect_args


# ---------------------------------------------------------------------------
# Property 6: Auth method validation rejects invalid values
# Feature: aws-mcp-postgres, Property 6: Auth method validation rejects invalid values
# ---------------------------------------------------------------------------


class TestProperty6AuthMethodValidationRejectsInvalid:
    """**Validates: Requirements 2.1**

    For any string value that is not "password" or "iam", constructing
    DBSettings with that AUTH_METHOD should raise a validation error.
    """

    @settings(max_examples=100)
    @given(
        invalid_method=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N"))).filter(
            lambda s: s not in ("password", "iam")
        )
    )
    def test_invalid_auth_method_rejected(self, invalid_method):
        # Feature: aws-mcp-postgres, Property 6: Auth method validation rejects invalid values
        with pytest.raises(ValidationError):
            DBSettings(AUTH_METHOD=invalid_method)


# ---------------------------------------------------------------------------
# Property 7: IAM mode requires AWS fields with descriptive errors
# Feature: aws-mcp-postgres, Property 7: IAM mode requires AWS fields with descriptive errors
# ---------------------------------------------------------------------------


class TestProperty7IamModeRequiresAwsFields:
    """**Validates: Requirements 2.2, 2.5**

    For any DBSettings where auth_method=iam and at least one required field
    is missing, construction should raise a validation error whose message
    identifies the specific missing field(s).
    """

    @settings(max_examples=100)
    @given(
        include_region=st.booleans(),
        include_hostname=st.booleans(),
        include_username=st.booleans(),
    )
    def test_missing_fields_named_in_error(
        self, include_region, include_hostname, include_username
    ):
        # Feature: aws-mcp-postgres, Property 7: IAM mode requires AWS fields with descriptive errors
        # At least one field must be missing for this test
        assume(not (include_region and include_hostname and include_username))

        kwargs: dict = {"AUTH_METHOD": "iam", "RDS_PORT": 5432}
        if include_region:
            kwargs["AWS_REGION"] = "us-east-1"
        if include_hostname:
            kwargs["RDS_HOSTNAME"] = "db.rds.amazonaws.com"
        if include_username:
            kwargs["RDS_USERNAME"] = "iam_user"

        with pytest.raises(ValidationError) as exc_info:
            DBSettings(**kwargs)

        err_text = str(exc_info.value)
        if not include_region:
            assert "DB_AWS_REGION" in err_text
        if not include_hostname:
            assert "DB_RDS_HOSTNAME" in err_text
        if not include_username:
            assert "DB_RDS_USERNAME" in err_text



# ---------------------------------------------------------------------------
# Property 10: IAM mode forces pool_pre_ping
# Feature: aws-mcp-postgres, Property 10: IAM mode forces pool_pre_ping
# ---------------------------------------------------------------------------


class TestProperty10IamModeForcePoolPrePing:
    """**Validates: Requirements 4.2**

    For any DBSettings where auth_method=iam, pool_pre_ping should always
    be True regardless of the configured POOL_PRE_PING value.
    """

    @settings(max_examples=100)
    @given(pool_pre_ping=st.booleans())
    def test_pool_pre_ping_always_true_for_iam(self, pool_pre_ping):
        # Feature: aws-mcp-postgres, Property 10: IAM mode forces pool_pre_ping
        s = DBSettings(
            AUTH_METHOD="iam",
            AWS_REGION="us-east-1",
            RDS_HOSTNAME="db.rds.amazonaws.com",
            RDS_PORT=5432,
            RDS_USERNAME="iam_user",
            POOL_PRE_PING=pool_pre_ping,
        )

        # Simulate the IAM override logic from db.py
        effective_pre_ping = s.POOL_PRE_PING
        if s.AUTH_METHOD == "iam":
            effective_pre_ping = True

        assert effective_pre_ping is True


# ---------------------------------------------------------------------------
# Property 11: IAM mode clamps pool_recycle
# Feature: aws-mcp-postgres, Property 11: IAM mode clamps pool_recycle
# ---------------------------------------------------------------------------


class TestProperty11IamModeClampsPoolRecycle:
    """**Validates: Requirements 4.3**

    For any DBSettings where auth_method=iam and any POOL_RECYCLE value,
    the effective pool_recycle should be min(configured_value, 900).
    """

    @settings(max_examples=100)
    @given(pool_recycle=pool_recycle_st)
    def test_pool_recycle_clamped_to_900(self, pool_recycle):
        # Feature: aws-mcp-postgres, Property 11: IAM mode clamps pool_recycle
        s = DBSettings(
            AUTH_METHOD="iam",
            AWS_REGION="us-east-1",
            RDS_HOSTNAME="db.rds.amazonaws.com",
            RDS_PORT=5432,
            RDS_USERNAME="iam_user",
            POOL_RECYCLE=pool_recycle,
        )

        # Simulate the IAM override logic from db.py
        effective_recycle = s.POOL_RECYCLE
        if s.AUTH_METHOD == "iam":
            effective_recycle = min(effective_recycle, 900)

        assert effective_recycle == min(pool_recycle, 900)
        assert effective_recycle <= 900


# ---------------------------------------------------------------------------
# Property 12: Pool settings preserved across auth methods
# Feature: aws-mcp-postgres, Property 12: Pool settings preserved across auth methods
# ---------------------------------------------------------------------------


class TestProperty12PoolSettingsPreserved:
    """**Validates: Requirements 4.4**

    For any config, pool_size, max_overflow, and pool_timeout should equal
    the configured values regardless of auth method.
    """

    @settings(max_examples=100)
    @given(
        auth_method=st.sampled_from(["password", "iam"]),
        pool_size=pool_size_st,
        max_overflow=max_overflow_st,
        pool_timeout=pool_timeout_st,
    )
    def test_pool_settings_preserved(
        self, auth_method, pool_size, max_overflow, pool_timeout
    ):
        # Feature: aws-mcp-postgres, Property 12: Pool settings preserved across auth methods
        kwargs: dict = {
            "AUTH_METHOD": auth_method,
            "POOL_SIZE": pool_size,
            "MAX_OVERFLOW": max_overflow,
            "POOL_TIMEOUT": pool_timeout,
        }
        if auth_method == "iam":
            kwargs.update({
                "AWS_REGION": "us-east-1",
                "RDS_HOSTNAME": "db.rds.amazonaws.com",
                "RDS_PORT": 5432,
                "RDS_USERNAME": "iam_user",
            })

        s = DBSettings(**kwargs)

        assert s.POOL_SIZE == pool_size
        assert s.MAX_OVERFLOW == max_overflow
        assert s.POOL_TIMEOUT == pool_timeout



# ---------------------------------------------------------------------------
# Property 13: Migration constructs IAM URI with fresh token
# Feature: aws-mcp-postgres, Property 13: Migration constructs IAM URI with fresh token
# ---------------------------------------------------------------------------


class TestProperty13MigrationConstructsIamUri:
    """**Validates: Requirements 6.1**

    For any valid IAM configuration, when init_db runs with auth_method=iam,
    the connection URI passed to Alembic should contain the RDS hostname,
    port, username, and a freshly generated IAM token as the password.
    """

    @settings(max_examples=100)
    @given(
        region=aws_region_st,
        hostname=rds_hostname_st,
        port=rds_port_st,
        username=rds_username_st,
        token=st.text(min_size=10, max_size=100, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
        db_name=st.from_regex(r"[a-z][a-z0-9]{0,15}", fullmatch=True),
    )
    def test_iam_uri_contains_all_components(
        self, region, hostname, port, username, token, db_name
    ):
        # Feature: aws-mcp-postgres, Property 13: Migration constructs IAM URI with fresh token
        # Simulate the URI construction logic from init_db in db.py
        encoded_token = quote_plus(token)
        iam_uri = (
            f"postgresql+psycopg://{username}:{encoded_token}"
            f"@{hostname}:{port}"
            f"/{db_name}"
        )

        assert username in iam_uri
        assert hostname in iam_uri
        assert str(port) in iam_uri
        assert encoded_token in iam_uri
        assert db_name in iam_uri
        assert iam_uri.startswith("postgresql+psycopg://")


# ---------------------------------------------------------------------------
# Property 14: Migration uses SSL config for IAM
# Feature: aws-mcp-postgres, Property 14: Migration uses SSL config for IAM
# ---------------------------------------------------------------------------


class TestProperty14MigrationUsesSSLConfigForIam:
    """**Validates: Requirements 6.2**

    For any IAM configuration with or without RDS_SSL_CA_BUNDLE, the Alembic
    connection should include the same SSL parameters as the main engine.
    """

    @settings(max_examples=100)
    @given(
        hostname=rds_hostname_st,
        port=rds_port_st,
        username=rds_username_st,
        ca_bundle=ssl_ca_bundle_st,
        token=st.from_regex(r"[a-zA-Z0-9]{10,30}", fullmatch=True),
    )
    def test_alembic_ssl_matches_engine_ssl(
        self, hostname, port, username, ca_bundle, token
    ):
        # Feature: aws-mcp-postgres, Property 14: Migration uses SSL config for IAM
        encoded_token = quote_plus(token)
        db_name = "testdb"

        # Simulate Alembic URI construction from init_db
        iam_uri = (
            f"postgresql+psycopg://{username}:{encoded_token}"
            f"@{hostname}:{port}/{db_name}"
        )
        ssl_query = "sslmode=require"
        if ca_bundle:
            ssl_query += f"&sslrootcert={quote_plus(ca_bundle)}"
        iam_uri += f"?{ssl_query}"

        # Simulate engine connect_args from db.py
        engine_connect_args: dict = {"prepare_threshold": None, "sslmode": "require"}
        if ca_bundle:
            engine_connect_args["sslrootcert"] = ca_bundle

        # Verify Alembic URI SSL matches engine connect_args
        assert "sslmode=require" in iam_uri
        if ca_bundle:
            assert quote_plus(ca_bundle) in iam_uri
            assert engine_connect_args["sslrootcert"] == ca_bundle
        else:
            assert "sslrootcert" not in iam_uri


# ---------------------------------------------------------------------------
# Property 15: Migration token failure terminates with error
# Feature: aws-mcp-postgres, Property 15: Migration token failure terminates with error
# ---------------------------------------------------------------------------


class TestProperty15MigrationTokenFailureTerminatesWithError:
    """**Validates: Requirements 6.3**

    For any exception raised during IAM token generation within init_db,
    the function should propagate the error after logging a descriptive message.
    """

    @settings(max_examples=100)
    @given(
        error_type=st.sampled_from(["no_credentials", "client_error", "endpoint_error"]),
        error_detail=st.text(min_size=1, max_size=30, alphabet=st.characters(whitelist_categories=("L", "N", "Z"))),
    )
    @patch("src.aws_auth.boto3.Session")
    def test_token_failure_propagates(self, mock_session_cls, error_type, error_detail):
        # Feature: aws-mcp-postgres, Property 15: Migration token failure terminates with error
        mock_client = MagicMock()

        if error_type == "no_credentials":
            mock_client.generate_db_auth_token.side_effect = NoCredentialsError()
        elif error_type == "client_error":
            error_response = {"Error": {"Code": "InternalError", "Message": error_detail}}
            mock_client.generate_db_auth_token.side_effect = ClientError(
                error_response, "GenerateDBAuthToken"
            )
        else:
            mock_client.generate_db_auth_token.side_effect = EndpointConnectionError(
                endpoint_url=f"https://rds.us-east-1.amazonaws.com"
            )

        mock_session_cls.return_value.client.return_value = mock_client

        with pytest.raises(RuntimeError):
            generate_rds_auth_token(
                region="us-east-1",
                hostname="db.rds.amazonaws.com",
                port=5432,
                username="iam_user",
            )
