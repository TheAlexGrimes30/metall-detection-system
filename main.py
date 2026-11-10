import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI

from src.auth import get_current_user, get_secret_key, router
from src.db.db_session import close_db, global_init
from src.db.models.user import User


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_secret_key()
    await global_init(os.getenv('DATABASE_URL', ''))
    try:
        yield
    finally:
        await close_db()


app = FastAPI(lifespan=lifespan)
app.include_router(router)


@app.post('/initialize')
async def initialize(user: Annotated[User, Depends(get_current_user)]):
    return {'message': 'Initialized'}

