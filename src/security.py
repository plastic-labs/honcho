import os
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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
        raise HTTPException(status_code=401, detail="Invalid access token")
    return {"message": "OK"}
