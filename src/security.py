import datetime
import logging
import os
import secrets
from collections import OrderedDict
from typing import Annotated, Optional

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud
from src.dependencies import get_db

from .exceptions import AuthenticationException

logger = logging.getLogger(__name__)

USE_AUTH = os.getenv("USE_AUTH", "False").lower() == "true"
AUTH_JWT_SECRET = os.getenv("AUTH_JWT_SECRET", "")

security = HTTPBearer(
    auto_error=False,
)

# LRU cache for API keys
# Structure: {token: is_revoked}
API_KEY_CACHE = OrderedDict()
CACHE_MAX_SIZE = 10


def clear_api_key_cache():
    """Clear the API key cache when keys are revoked."""
    API_KEY_CACHE.clear()


def cache_api_key(token: str, is_revoked: bool):
    """Add an API key to the cache."""
    API_KEY_CACHE[token] = is_revoked
    API_KEY_CACHE.move_to_end(token)

    if len(API_KEY_CACHE) > CACHE_MAX_SIZE:
        API_KEY_CACHE.popitem(last=False)


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
    Hierarchy: app > user > (session / collection)

    All routers require at least the most tightly scoped parameter.
    Routes will accept a JWT with a scope higher in the hierarchy.

    Names shortened to minimize token size.
    """

    t: str = datetime.datetime.now().isoformat()
    ad: Optional[bool] = None
    ap: Optional[str] = None
    us: Optional[str] = None
    se: Optional[str] = None
    co: Optional[str] = None


# generate jwt token
# if secret key is empty, generate a random one
if not AUTH_JWT_SECRET:
    AUTH_JWT_SECRET = secrets.token_hex(32)
    print(f"\n    Generated secret key: {AUTH_JWT_SECRET}")


def rotate_jwt_secret(new_secret: str | None = None) -> str:
    """
    Rotate the JWT secret and return the new admin JWT.
    If no new secret is provided, generate a new one.
    """
    global AUTH_JWT_SECRET
    AUTH_JWT_SECRET = new_secret if new_secret else secrets.token_hex(32)

    # Clear the cache when rotating secrets
    clear_api_key_cache()

    return create_admin_jwt()


def create_admin_jwt() -> str:
    """Create a JWT for admin operations."""
    params = JWTParams(t="", ad=True)
    key = create_jwt(params)
    return key


def create_jwt(params: JWTParams) -> str:
    """Create a JWT token from the given parameters."""
    payload = {k: v for k, v in params.__dict__.items() if v is not None}
    return jwt.encode(payload, AUTH_JWT_SECRET.encode("utf-8"), algorithm="HS256")


async def verify_jwt(token: str, db: AsyncSession) -> JWTParams:
    """Verify a JWT token and return the decoded parameters."""
    # Check if key has been revoked using cache first
    if token in API_KEY_CACHE:
        is_revoked = API_KEY_CACHE[token]
        if is_revoked:
            raise AuthenticationException("Key is revoked")
    else:
        # If not in cache, check the database
        key = await crud.get_key(db, token)
        if key:
            # Cache the result
            cache_api_key(token, key.revoked)
            if key.revoked:
                raise AuthenticationException("Key is revoked")
        else:
            # Key not found in DB, cache as not revoked
            cache_api_key(token, False)

    params = JWTParams()
    try:
        decoded = jwt.decode(
            token, AUTH_JWT_SECRET.encode("utf-8"), algorithms=["HS256"]
        )
        if "t" in decoded:
            params.t = decoded["t"]
        if "ad" in decoded:
            params.ad = decoded["ad"]
        if "ap" in decoded:
            params.ap = decoded["ap"]
        if "us" in decoded:
            params.us = decoded["us"]
        if "se" in decoded:
            params.se = decoded["se"]
        if "co" in decoded:
            params.co = decoded["co"]
        return params
    except jwt.PyJWTError:
        raise AuthenticationException("Invalid JWT") from None


def require_auth(
    admin: Optional[bool] = None,
    app_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    collection_id: Optional[str] = None,
):
    """
    Generate a dependency that requires authentication for the given parameters.
    """

    async def auth_dependency(
        request: Request,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: AsyncSession = Depends(get_db),
    ):
        app_id_param = request.path_params.get(app_id) if app_id else None
        user_id_param = request.path_params.get(user_id) if user_id else None
        session_id_param = request.path_params.get(session_id) if session_id else None
        collection_id_param = (
            request.path_params.get(collection_id) if collection_id else None
        )

        return await auth(
            credentials=credentials,
            db=db,
            admin=admin,
            app_id=app_id_param,
            user_id=user_id_param,
            session_id=session_id_param,
            collection_id=collection_id_param,
        )

    return auth_dependency


async def auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: AsyncSession = Depends(get_db),
    admin: Optional[bool] = None,
    app_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    collection_id: Optional[str] = None,
) -> JWTParams:
    """Authenticate the given JWT and return the decoded parameters."""
    if not USE_AUTH:
        return JWTParams(t="", ad=True)
    if not credentials or not credentials.credentials:
        logger.warning("No access token provided")
        raise AuthenticationException("No access token provided")

    jwt_params = await verify_jwt(credentials.credentials, db)

    # based on api operation, verify api key based on that key's permissions
    if jwt_params.ad:
        return jwt_params
    if admin:
        raise AuthenticationException("Resource requires admin privileges")

    # Check if the JWT has direct access to the requested resource
    # For session or collection level access
    if session_id and jwt_params.se == session_id:
        return jwt_params
    if collection_id and jwt_params.co == collection_id:
        return jwt_params

    # For user level access - can access all sessions/collections under this user
    if user_id and jwt_params.us == user_id:
        return jwt_params

    # For app level access - can access all users/sessions/collections under this app
    if app_id and jwt_params.ap == app_id:
        return jwt_params

    if any([session_id, collection_id, user_id, app_id]):
        raise AuthenticationException("JWT not permissioned for this resource")

    # Route did not specify any parameters, so it should parse parameters itself
    return jwt_params
