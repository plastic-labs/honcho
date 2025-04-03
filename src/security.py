import logging
import os
from typing import Annotated

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .exceptions import AuthenticationException

logger = logging.getLogger(__name__)

USE_AUTH_SERVICE = os.getenv("USE_AUTH_SERVICE", "False").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "test")

security = HTTPBearer(
    auto_error=False,
)


async def auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    if not USE_AUTH_SERVICE:
        return True
    if not credentials or credentials.credentials != SECRET_KEY:
        logger.warning("Invalid access token attempt")
        raise AuthenticationException("Invalid access token")
    return {"message": "OK"}
