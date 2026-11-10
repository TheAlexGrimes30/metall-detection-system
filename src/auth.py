import os
from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from src.db.db_session import get_session
from src.db.models.user import User

router = APIRouter(prefix='/auth', tags=['auth'])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='auth/token')
password_hash = PasswordHash.recommended()
dummy_hash = password_hash.hash('unused-password-for-timing')
SessionDependency = Annotated[AsyncSession, Depends(get_session)]


def get_secret_key() -> str:
    key = os.getenv('JWT_SECRET_KEY', '')
    if len(key) < 32:
        raise RuntimeError('Set JWT_SECRET_KEY to a random secret of at least 32 characters')
    return key


class Token(BaseModel):
    access_token: str
    token_type: str = 'bearer'


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


def credentials_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='Invalid credentials',
        headers={'WWW-Authenticate': 'Bearer'},
    )


def create_access_token(user_id: int) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {'sub': str(user_id), 'iat': now, 'exp': now + timedelta(minutes=30)},
        get_secret_key(),
        algorithm='HS256',
    )


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)], session: SessionDependency,
) -> User:
    try:
        payload = jwt.decode(
            token, get_secret_key(), algorithms=['HS256'],
            options={'require': ['sub', 'exp', 'iat']},
        )
        user_id = int(payload['sub'])
    except (InvalidTokenError, ValueError, TypeError):
        raise credentials_error() from None
    user = await session.get(User, user_id)
    if user is None or user.is_deleted:
        raise credentials_error()
    return user


@router.post('/token', response_model=Token)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()], session: SessionDependency,
):
    user = await session.scalar(select(User).where(User.name == form.username))
    valid = await run_in_threadpool(
        password_hash.verify, form.password, user.password_hash if user else dummy_hash,
    )
    if not valid or user is None or user.is_deleted:
        raise credentials_error()
    return Token(access_token=create_access_token(user.id))


@router.get('/me', response_model=UserResponse)
async def me(user: Annotated[User, Depends(get_current_user)]):
    return user
