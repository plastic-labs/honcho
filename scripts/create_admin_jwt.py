import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import dotenv
import jwt

from src.security import JWTParams

dotenv.load_dotenv()

SECRET = os.getenv("AUTH_JWT_SECRET")

if not SECRET:
    raise ValueError("AUTH_JWT_SECRET is not set")


jwt_params = JWTParams(
    t="",
    ad=True,
)

payload = {k: v for k, v in jwt_params.__dict__.items() if v is not None}

print(jwt.encode(payload, SECRET, algorithm="HS256"))
