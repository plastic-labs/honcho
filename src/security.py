import datetime
import logging
from typing import Annotated

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.config import settings

from .exceptions import AuthenticationException

logger = logging.getLogger(__name__)

security = HTTPBearer(
    auto_error=False,
)


#
#  jwt params
#  all optional, used to produce tokens valid for different routes
#  hierarchy: app > user > ( session / collection )
#  routes that involve a 'name' parameter require permissions for the parent object
#  name routes are considered 'queries' as names are mutable properties
#
#  note: add routes without parameters that assume the most immediately scoped key is providing
#
class JWTParams(BaseModel):
    """
    JWT parameters used to produce tokens valid for different routes.
    Workspaces are the top level of the hierarchy -- a workspace key will
    give access to all peers/sessions/collections in that workspace.

    A session key will allow the listing and creation of messages in
    that session.

    A peer key will allow the listing and creation of peer-level messages
    and querying the peer's dialectic endpoint.

    Names shortened to minimize token size. Timestamp is included
    so that many unique tokens can be generated for the same resource.
    Note that the timestamp itself is not used for security, and can
    be omitted, such as when Honcho generates the initial admin JWT.

    Fields (all optional other than `t`):

    `t`: a string timestamp of when the JWT was created
    `exp`: a string timestamp of when the JWT expires (optional)
    `ad`: a boolean flag indicating if the JWT is an admin JWT
    `w`: (string) workspace name
    `p`: (string) peer name
    `s`: (string) session name
    """

    t: str = datetime.datetime.now().isoformat()
    exp: str | None = None
    ad: bool | None = None
    w: str | None = None
    p: str | None = None
    s: str | None = None


def create_admin_jwt() -> str:
    """Create a JWT for admin operations."""
    params = JWTParams(t="", ad=True)
    key = create_jwt(params)
    return key


def create_jwt(params: JWTParams) -> str:
    """Create a JWT from the given parameters."""
    payload = {k: v for k, v in params.__dict__.items() if v is not None}
    if not settings.AUTH.JWT_SECRET:
        raise ValueError("AUTH_JWT_SECRET is not set, cannot create JWT.")
    return jwt.encode(
        payload, settings.AUTH.JWT_SECRET.encode("utf-8"), algorithm="HS256"
    )


async def verify_jwt(token: str) -> JWTParams:
    """Verify a JWT and return the decoded parameters."""

    params = JWTParams()
    try:
        if not settings.AUTH.JWT_SECRET:
            raise ValueError("AUTH_JWT_SECRET is not set, cannot verify JWT.")
        decoded = jwt.decode(
            token, settings.AUTH.JWT_SECRET.encode("utf-8"), algorithms=["HS256"]
        )
        if "t" in decoded:
            params.t = decoded["t"]
        if "exp" in decoded:
            params.exp = decoded["exp"]
            if (
                params.exp
                and datetime.datetime.fromisoformat(params.exp)
                < datetime.datetime.now()
            ):
                raise AuthenticationException("JWT expired")
        if "ad" in decoded:
            params.ad = decoded["ad"]
        if "w" in decoded:
            params.w = decoded["w"]
        if "p" in decoded:
            params.p = decoded["p"]
        if "s" in decoded:
            params.s = decoded["s"]
        return params
    except jwt.PyJWTError:
        raise AuthenticationException("Invalid JWT") from None


def require_auth(
    admin: bool | None = None,
    workspace_name: str | None = None,
    peer_name: str | None = None,
    session_name: str | None = None,
):
    """
    Generate a dependency that requires authentication for the given parameters.
    """

    async def auth_dependency(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
    ):
        workspace_name_param = (
            request.path_params.get(workspace_name)
            or request.query_params.get(workspace_name)
            if workspace_name
            else None
        )
        peer_name_param = (
            request.path_params.get(peer_name) or request.query_params.get(peer_name)
            if peer_name
            else None
        )
        session_name_param = (
            request.path_params.get(session_name)
            or request.query_params.get(session_name)
            if session_name
            else None
        )

        return await auth(
            credentials=credentials,
            admin=admin,
            workspace_name=workspace_name_param,
            peer_name=peer_name_param,
            session_name=session_name_param,
        )

    return auth_dependency


async def auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    admin: bool | None = None,
    workspace_name: str | None = None,
    peer_name: str | None = None,
    session_name: str | None = None,
) -> JWTParams:
    """Authenticate the given JWT and return the decoded parameters."""
    if not settings.AUTH.USE_AUTH:
        return JWTParams(t="", ad=True)
    if not credentials or not credentials.credentials:
        logger.warning("No access token provided")
        raise AuthenticationException("No access token provided")

    jwt_params = await verify_jwt(credentials.credentials)

    # based on api operation, verify api key based on that key's permissions
    if jwt_params.ad:
        return jwt_params
    if admin:
        raise AuthenticationException("Resource requires admin privileges")

    # For session level access
    if session_name and jwt_params.s == session_name:
        if workspace_name and jwt_params.w != workspace_name:
            raise AuthenticationException("JWT not permissioned for this resource")
        return jwt_params

    # For peer level access
    if peer_name and jwt_params.p == peer_name:
        if workspace_name and jwt_params.w != workspace_name:
            raise AuthenticationException("JWT not permissioned for this resource")
        return jwt_params

    # For workspace level access - can access all peers/sessions under this workspace
    if workspace_name and jwt_params.w == workspace_name:
        return jwt_params

    if any([session_name, peer_name, workspace_name]):
        raise AuthenticationException("JWT not permissioned for this resource")

    # Route did not specify any parameters, so it should parse parameters itself
    return jwt_params
