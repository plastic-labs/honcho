import datetime
import logging
from typing import Annotated

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from src.config import settings
from src.utils.formatting import parse_datetime_iso, utc_now_iso

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

    t: str = Field(default_factory=utc_now_iso)
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


def scope_requires_workspace(
    *, peer: str | None, session: str | None, workspace: str | None
) -> bool:
    """Return whether a peer- or session-scoped claim lacks its parent workspace.

    A peer or session scope is meaningless without a workspace: the route-level
    check cannot rule out cross-workspace use (a ``{p: "alice"}`` token would
    match ``alice`` in any workspace). Truthiness-based so empty-string claims
    count as absent. Shared by `verify_jwt` (the token-shape invariant) and the
    keys API (the creation-time guard) so the two rules cannot drift apart.

    Args:
        peer: The peer claim, if any.
        session: The session claim, if any.
        workspace: The workspace claim, if any.

    Returns:
        True when a peer/session scope is present but the workspace is not.
    """
    return bool(peer or session) and not workspace


def verify_jwt(token: str) -> JWTParams:
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
            if params.exp:
                exp_time = parse_datetime_iso(params.exp)
                current_time = datetime.datetime.now(datetime.timezone.utc)
                if exp_time < current_time:
                    raise AuthenticationException("JWT expired")
        if "ad" in decoded:
            params.ad = decoded["ad"]
        # Normalize empty-string scope claims to None so a blank `w`/`p`/`s`
        # cannot masquerade as a present claim in the checks below.
        if "w" in decoded:
            params.w = decoded["w"] or None
        if "p" in decoded:
            params.p = decoded["p"] or None
        if "s" in decoded:
            params.s = decoded["s"] or None
        # Token-shape invariant: a peer- or session-scoped token MUST also
        # carry its parent workspace, otherwise the route-level check cannot
        # rule out cross-workspace use.
        if scope_requires_workspace(
            peer=params.p, session=params.s, workspace=params.w
        ):
            raise AuthenticationException(
                "Invalid JWT scope: peer/session token missing workspace"
            )
        return params
    except jwt.PyJWTError:
        raise AuthenticationException("Invalid JWT") from None


def require_auth(
    admin: bool | None = None,
    workspace_name: str | None = None,
    peer_name: str | None = None,
    session_name: str | None = None,
    allow_member_read: bool = False,
):
    """
    Generate a dependency that requires authentication for the given parameters.

    Set `allow_member_read=True` on read-only session routes to additionally
    grant access to peer-scoped keys whose peer is an active member of the
    session. Never set it on routes that mutate state.
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
            allow_member_read=allow_member_read,
        )

    # Tag the closure so route-policy tests can introspect which routes opt into
    # member read without re-deriving it from HTTP method (an unreliable
    # read/write signal here — some read routes use POST for a richer body).
    auth_dependency.honcho_allow_member_read = allow_member_read  # pyright: ignore[reportFunctionMemberAccess]

    return auth_dependency


async def auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    admin: bool | None = None,
    workspace_name: str | None = None,
    peer_name: str | None = None,
    session_name: str | None = None,
    allow_member_read: bool = False,
) -> JWTParams:
    """Authenticate the given JWT and return the decoded parameters."""
    if not settings.AUTH.USE_AUTH:
        return JWTParams(t="", ad=True)
    if not credentials or not credentials.credentials:
        logger.warning("No access token provided")
        raise AuthenticationException("No access token provided")

    jwt_params = verify_jwt(credentials.credentials)

    # Authorize by the token's narrowest scope, not by the route's. A
    # narrower-than-workspace token must NOT fall back to workspace access:
    # `{w: ws, p: alice}` may only act on `alice`, never on a sibling peer.
    if jwt_params.ad:
        return jwt_params
    if admin:
        raise AuthenticationException("Resource requires admin privileges")

    if not any([session_name, peer_name, workspace_name]):
        # Self-authorizing routes decode the token here and compare the claims
        # against body/path data inside the handler. This is needed for routes
        # whose resource identifier is not available to require_auth().
        return jwt_params

    # Every scoped, non-admin path requires the token's workspace to match the
    # route's. Check it once here so no individual branch below can forget it
    # and silently re-open cross-workspace access (the bug this module fixes).
    if workspace_name and jwt_params.w != workspace_name:
        raise AuthenticationException("JWT not permissioned for this resource")

    if jwt_params.s is not None:
        # Session-scoped token: confined to its own session. It gets no
        # cross-scope access to peer routes.
        if not session_name or jwt_params.s != session_name:
            raise AuthenticationException("JWT not permissioned for this resource")
        return jwt_params

    if jwt_params.p is not None:
        # Peer-scoped token: its own peer routes...
        if peer_name and jwt_params.p == peer_name:
            return jwt_params
        # ...plus read-only access to the sessions the peer is a member of.
        # Gated on `allow_member_read` so only read routes opt in; writes stay
        # denied. Requires the route's workspace so the membership lookup is
        # scoped (every session route declares workspace_name); the workspace
        # match itself was already verified above.
        if allow_member_read and session_name and workspace_name:
            # Lazy imports avoid an import cycle with the crud/db layers and
            # keep this DB round-trip off the common (same-scope) auth path.
            from src.crud.session import is_peer_in_session
            from src.dependencies import tracked_db

            # Membership is read on a separate committed-only (read_only)
            # connection, so a peer added to the session in a not-yet-committed
            # transaction reads as a non-member: writes must commit before a
            # member-scoped read. Fails closed.
            async with tracked_db(
                "auth.is_peer_in_session", read_only=True
            ) as member_db:
                is_member = await is_peer_in_session(
                    member_db, workspace_name, session_name, jwt_params.p
                )
            if is_member:
                return jwt_params
        raise AuthenticationException("JWT not permissioned for this resource")

    if jwt_params.w is not None:
        # Workspace tokens reach any route inside their workspace (the workspace
        # match was verified above). Routes without a declared workspace (e.g.
        # POST /v3/workspaces) self-authorize by reading jwt_params.w themselves.
        return jwt_params

    raise AuthenticationException("JWT not permissioned for this resource")
