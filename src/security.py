import logging
import os
from typing import Annotated, Optional

import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .exceptions import AuthenticationException

logger = logging.getLogger(__name__)

USE_AUTH = os.getenv("USE_AUTH", "False").lower() == "true"
AUTH_JWT_SECRET = os.getenv("AUTH_JWT_SECRET", "")

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
    Hierarchy: app > user > (session / collection)

    All routers require at least the most tightly scoped parameter.
    Routes will accept a JWT with a scope higher in the hierarchy.

    Names shortened to minimize token size.
    """

    ad: Optional[bool] = None
    ap: Optional[str] = None
    us: Optional[str] = None
    se: Optional[str] = None
    co: Optional[str] = None


# generate jwt token
# if secret key is empty, generate a random one
if not AUTH_JWT_SECRET:
    import secrets

    AUTH_JWT_SECRET = secrets.token_hex(32)
    print(f"Generated secret key: {AUTH_JWT_SECRET}")


def create_admin_jwt() -> str:
    params = JWTParams()
    params.ad = True
    return create_jwt(params)


def create_jwt(params: JWTParams) -> str:
    return jwt.encode(
        params.__dict__, AUTH_JWT_SECRET.encode("utf-8"), algorithm="HS256"
    )


def verify_jwt(token: str) -> JWTParams:
    params = JWTParams()
    try:
        decoded = jwt.decode(
            token, AUTH_JWT_SECRET.encode("utf-8"), algorithms=["HS256"]
        )
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
        print(f"Decoded JWT: {params}")
        return params
    except jwt.PyJWTError:
        print("Invalid JWT")
        raise AuthenticationException("Invalid JWT") from None


def require_auth(
    admin: Optional[bool] = None,
    app_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    collection_id: Optional[str] = None,
):
    async def auth_dependency(
        request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        app_id_param = request.path_params.get(app_id) if app_id else None
        user_id_param = request.path_params.get(user_id) if user_id else None
        session_id_param = request.path_params.get(session_id) if session_id else None
        collection_id_param = (
            request.path_params.get(collection_id) if collection_id else None
        )

        return await auth(
            credentials=credentials,
            admin=admin,
            app_id=app_id_param,
            user_id=user_id_param,
            session_id=session_id_param,
            collection_id=collection_id_param,
        )

    return auth_dependency


async def auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    admin: Optional[bool] = None,
    app_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    collection_id: Optional[str] = None,
):
    if not USE_AUTH:
        return True
    if not credentials or not credentials.credentials:
        logger.warning("No access token provided")
        raise AuthenticationException("No access token provided")
    jwt_params = verify_jwt(credentials.credentials)
    if not jwt_params:
        logger.warning("Invalid access token attempt")
        raise AuthenticationException("Invalid access token")

    # based on api operation, verify api key based on that key's permissions
    if jwt_params.ad:
        return {"message": "OK"}
    if admin:
        raise AuthenticationException("Resource requires admin privileges")
    if app_id and jwt_params.ap == app_id:
        return {"message": "OK"}
    if user_id and jwt_params.us == user_id:
        return {"message": "OK"}
    if session_id and jwt_params.se == session_id:
        return {"message": "OK"}
    if collection_id and jwt_params.co == collection_id:
        return {"message": "OK"}
    raise AuthenticationException("JWT not permissioned for this resource")
