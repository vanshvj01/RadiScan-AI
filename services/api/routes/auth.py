"""Registration, login, refresh, profile."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from services.api.core.security import (
    create_token,
    current_user,
    decode_token,
    hash_password,
    verify_password,
)
from services.api.core.settings import get_settings
from services.api.db.models import AuditLog, Role, User
from services.api.db.session import get_db
from services.api.schemas.schemas import RefreshRequest, TokenPair, UserCreate, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


def _audit(db: Session, user_id: int | None, action: str, resource: str, request: Request) -> None:
    db.add(
        AuditLog(
            user_id=user_id,
            action=action,
            resource=resource,
            ip_address=request.client.host if request.client else None,
        )
    )
    db.commit()


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, request: Request, db: Session = Depends(get_db)) -> User:
    exists = (
        db.query(User)
        .filter((User.username == payload.username) | (User.email == payload.email))
        .first()
    )
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "Username or email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=Role(payload.role),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    _audit(db, user.id, "user.register", f"user:{user.id}", request)
    return user


@router.post("/login", response_model=TokenPair)
def login(
    request: Request,
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> TokenPair:
    user = db.query(User).filter(User.username == form.username).first()
    # Verify against a dummy hash even when the user is absent, so response time
    # does not reveal whether a username exists.
    hashed = user.hashed_password if user else "$2b$12$" + "x" * 53
    ok = verify_password(form.password, hashed) if user else False

    if not user or not ok or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password")

    _audit(db, user.id, "user.login", f"user:{user.id}", request)
    return TokenPair(
        access_token=create_token(user.username, "access", {"role": user.role.value}),
        refresh_token=create_token(user.username, "refresh"),
        expires_in=settings.access_token_minutes * 60,
    )


@router.post("/refresh", response_model=TokenPair)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)) -> TokenPair:
    claims = decode_token(payload.refresh_token, expected_type="refresh")
    user = db.query(User).filter(User.username == claims["sub"]).first()
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer active")
    return TokenPair(
        access_token=create_token(user.username, "access", {"role": user.role.value}),
        refresh_token=create_token(user.username, "refresh"),
        expires_in=settings.access_token_minutes * 60,
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> User:
    return user
