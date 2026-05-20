"""
auth_service.py
---------------
JWT + password hashing utilities.

Install: pip install python-jose[cryptography] passlib[bcrypt]
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session as DBSession
from dotenv import load_dotenv
import os

from app.database import get_db
from app.models.user import User, UserRole

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
SECRET_KEY            = os.getenv("SECRET_KEY",            "CHANGE_ME_IN_PRODUCTION")
REFRESH_SECRET_KEY    = os.getenv("REFRESH_SECRET_KEY",    "CHANGE_ME_REFRESH_IN_PRODUCTION")
ALGORITHM             = "HS256"
ACCESS_TOKEN_EXPIRE   = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES",  "15"))   # 15 min
REFRESH_TOKEN_EXPIRE  = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS",    "7"))    # 7 days
REFRESH_COOKIE_NAME   = "autops_refresh"

# ── Password hashing ───────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

def hash_password(password: str) -> str:
    # bcrypt max is 72 bytes
    return pwd_context.hash(password[:72])

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain[:72], hashed)


# ── JWT helpers ────────────────────────────────────────────────────────────
def create_access_token(user_id: int, role: str) -> str:
    expire  = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE)
    payload = {"sub": str(user_id), "role": role, "exp": expire, "type": "access"}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    expire  = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE)
    payload = {"sub": str(user_id), "exp": expire, "type": "refresh"}
    return jwt.encode(payload, REFRESH_SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "access":
            raise JWTError("Wrong token type")
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def decode_refresh_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, REFRESH_SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise JWTError("Wrong token type")
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )


# ── Refresh cookie helpers ─────────────────────────────────────────────────
def set_refresh_cookie(response, token: str):
    response.set_cookie(
        key      = REFRESH_COOKIE_NAME,
        value    = token,
        httponly = True,          # JS cannot read this
        secure   = False,         # set True in production (HTTPS)
        samesite = "lax",         # lax works for localhost dev (strict blocks cross-port)
        max_age  = REFRESH_TOKEN_EXPIRE * 24 * 60 * 60,  # seconds
        path     = "/api/auth",   # only sent to auth endpoints
    )


def clear_refresh_cookie(response):
    response.delete_cookie(
        key      = REFRESH_COOKIE_NAME,
        path     = "/api/auth",
        httponly = True,
        samesite = "lax",
    )


# ── FastAPI dependencies ───────────────────────────────────────────────────
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: DBSession = Depends(get_db),
) -> User:
    """Dependency — extracts and validates access token from Authorization header."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    user_id = int(payload["sub"])

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    return user


def get_current_active_user(user: User = Depends(get_current_user)) -> User:
    """Same as get_current_user — alias for clarity."""
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Dependency — only allows admin role."""
    if user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user