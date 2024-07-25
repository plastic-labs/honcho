import os
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

USE_AUTH_SERVICE = os.getenv("USE_AUTH_SERVICE", "False").lower() == "true"
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")

security = HTTPBearer(
    auto_error=False,
)


async def auth(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    if not USE_AUTH_SERVICE:
        return True
    print(credentials)
    if not credentials or credentials.credentials != "test":
        raise HTTPException(status_code=401, detail="Invalid access token")
    # payload = {"token": token}
    # res = httpx.get(
    #     f"{AUTH_SERVICE_URL}/validate",
    #     params=payload,
    # )
    # data = res.json()
    return {"message": "OK"}

    # return {"scheme": credentials.scheme, "token": credentials.credentials}
