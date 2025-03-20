import logging
import os
import jwt
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .exceptions import AuthenticationException

logger = logging.getLogger(__name__)

USE_AUTH_SERVICE = os.getenv("USE_AUTH_SERVICE", "False").lower() == "true"
AUTH_JWT_SECRET = os.getenv("AUTH_JWT_SECRET", "")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")

security = HTTPBearer(
    auto_error=False,
)

# generate jwt token
# if secret key is empty, generate a random one
if not AUTH_JWT_SECRET:
    import secrets
    AUTH_JWT_SECRET = secrets.token_hex(32)
    print(f"Generated secret key: {AUTH_JWT_SECRET}")

def create_jwt():
    return jwt.encode({"key":ADMIN_KEY}, AUTH_JWT_SECRET.encode('utf-8'), algorithm="HS256")

def verify_jwt(token: str):
    try:
        decoded = jwt.decode(token, AUTH_JWT_SECRET.encode('utf-8'), algorithms=["HS256"])
        return decoded["key"]
    except jwt.PyJWTError:
        print("Invalid JWT")
        raise AuthenticationException("Invalid JWT")

async def auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    if not USE_AUTH_SERVICE:
        return True
    if not credentials or not credentials.credentials:
        logger.warning("No access token provided")
        raise AuthenticationException("No access token provided")
    api_key = verify_jwt(credentials.credentials)
    if not api_key:
        logger.warning("Invalid access token attempt")
        raise AuthenticationException("Invalid access token")

    # based on api operation, verify api key based on that key's permissions
    if api_key == ADMIN_KEY:
        return {"message": "OK"}
    else:
        raise AuthenticationException("TODO")
