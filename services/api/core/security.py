"""Authentication. No mock users, no DEBUG bypass.

The v1 API injected a hardcoded ``MockUser(id=1)`` whenever ``DEBUG=True`` and
raised an unconditional 401 otherwise, so the upload endpoint was either wide
open or unusable. Here every protected route depends on :func:`current_user`,
which validates a real JWT against a real row. There is exactly one way in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from services.api.core.settings import get_settings
from services.api.db.models import Role, User
from services.api.db.session import get_db

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth/login")

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def _encode(raw: str) -> bytes:
    """bcrypt silently ignores everything past 72 bytes, so truncate explicitly.

    Doing it here rather than relying on the library means a 200-character
    passphrase behaves identically at hash time and verify time. (passlib is
    deliberately not used: its bcrypt backend raises on 4.x, and it is
    unmaintained.)
    """
    return raw.encode("utf-8")[:72]


def hash_password(raw: str) -> str:
    return bcrypt.hashpw(_encode(raw), bcrypt.gensalt()).decode("utf-8")


def verify_password(raw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_encode(raw), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_token(subject: str, kind: str = "access", extra: dict[str, Any] | None = None) -> str:
    now = datetime.now(timezone.utc)
    expires = now + (
        timedelta(minutes=settings.access_token_minutes)
        if kind == "access"
        else timedelta(days=settings.refresh_token_days)
    )
    payload = {"sub": subject, "type": kind, "iat": now, "exp": expires, **(extra or {})}
    return jwt.encode(payload, settings.resolved_secret(), algorithm=settings.algorithm)


def decode_token(token: str, expected_type: str = "access") -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.resolved_secret(), algorithms=[settings.algorithm])
    except JWTError as exc:
        raise CREDENTIALS_ERROR from exc
    if payload.get("type") != expected_type:
        raise CREDENTIALS_ERROR
    return payload


def current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        raise CREDENTIALS_ERROR
    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise CREDENTIALS_ERROR
    return user


def require_role(*roles: Role):
    """Route dependency enforcing role-based access control."""

    def dependency(user: User = Depends(current_user)) -> User:
        if roles and user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {', '.join(r.value for r in roles)}",
            )
        return user

    return dependency
