"""AWS credential provider for RDS IAM authentication.

Generates short-lived IAM authentication tokens for connecting to
AWS RDS PostgreSQL instances using IAM-based authentication.
"""

import logging

import boto3
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
    ProfileNotFound,
)

logger = logging.getLogger(__name__)


def generate_rds_auth_token(
    region: str,
    hostname: str,
    port: int,
    username: str,
    profile: str | None = None,
) -> str:
    """Generate a short-lived IAM auth token for RDS connection.

    Uses boto3's generate_db_auth_token to create a SigV4-signed token
    that can be used as a password for RDS IAM authentication.

    Args:
        region: AWS region of the RDS instance (e.g. "us-east-1").
        hostname: RDS instance hostname.
        port: RDS instance port.
        username: Database username configured for IAM auth.
        profile: Optional named AWS credentials profile.

    Returns:
        A short-lived IAM authentication token string.

    Raises:
        RuntimeError: If token generation fails due to missing credentials,
            insufficient permissions, or network issues.
    """
    try:
        session = boto3.Session(
            region_name=region,
            profile_name=profile,
        )
        client = session.client("rds", region_name=region)
        token: str = client.generate_db_auth_token(
            DBHostname=hostname,
            Port=port,
            DBUsername=username,
            Region=region,
        )
        return token
    except ProfileNotFound as exc:
        logger.error(
            "AWS profile not found for RDS IAM auth "
            "(region=%s, hostname=%s, username=%s, profile=%s): %s",
            region,
            hostname,
            username,
            profile,
            exc,
        )
        raise RuntimeError(
            f"AWS profile '{profile}' not found. "
            "Set DB_AWS_PROFILE to a valid profile or unset it to use default credentials."
        ) from exc
    except NoCredentialsError as exc:
        logger.error(
            "AWS credentials not found for RDS IAM auth "
            "(region=%s, hostname=%s, username=%s): %s",
            region,
            hostname,
            username,
            exc,
        )
        raise RuntimeError(
            "AWS credentials not found. Ensure an IAM role is attached, "
            "AWS environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) "
            "are set, or DB_AWS_PROFILE is configured with a valid profile."
        ) from exc
    except ClientError as exc:
        logger.error(
            "AWS client error during RDS IAM token generation "
            "(region=%s, hostname=%s, username=%s): %s",
            region,
            hostname,
            username,
            exc,
        )
        error_code = exc.response.get("Error", {}).get("Code", "")
        if "AccessDenied" in error_code or "Forbidden" in error_code:
            raise RuntimeError(
                "Access denied when generating RDS IAM auth token. "
                "Ensure the IAM policy grants the 'rds-db:connect' action "
                f"for the database user '{username}' on the target RDS resource."
            ) from exc
        raise RuntimeError(
            f"AWS client error during RDS IAM token generation: {exc}"
        ) from exc
    except EndpointConnectionError as exc:
        logger.error(
            "Cannot reach AWS endpoint for RDS IAM token generation "
            "(region=%s, hostname=%s, username=%s): %s",
            region,
            hostname,
            username,
            exc,
        )
        raise RuntimeError(
            f"Cannot connect to AWS STS/RDS endpoint in region '{region}'. "
            "Check network connectivity, VPC configuration, and ensure the "
            "region is correct."
        ) from exc
