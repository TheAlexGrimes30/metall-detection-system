from collections.abc import AsyncIterator

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase


class DBBase(DeclarativeBase):
    pass


engine: AsyncEngine | None = None
__factory: async_sessionmaker[AsyncSession] | None = None


def get_database_url(url: str) -> str:
    url = url.strip()
    if not url:
        raise ValueError('Задайте DATABASE_URL для подключения к PostgreSQL')
    parsed = make_url(url)
    if parsed.drivername not in {'postgres', 'postgresql', 'postgresql+asyncpg'}:
        raise ValueError('Поддерживается только PostgreSQL с драйвером asyncpg')
    return parsed.set(drivername='postgresql+asyncpg').render_as_string(hide_password=False)


async def global_init(url: str) -> None:
    global engine, __factory
    normalized_url = get_database_url(url)
    if __factory is not None:
        return
    new_engine = create_async_engine(normalized_url, echo=False)
    from . import __all_models  # noqa: F401
    try:
        async with new_engine.begin() as connection:
            await connection.run_sync(DBBase.metadata.create_all)
    except BaseException:
        await new_engine.dispose()
        raise
    engine = new_engine
    __factory = async_sessionmaker(engine, expire_on_commit=False)


async def close_db() -> None:
    global engine, __factory
    if engine is not None:
        await engine.dispose()
    engine = None
    __factory = None


def create_session() -> AsyncSession:
    if __factory is None:
        raise RuntimeError('База данных не инициализирована')
    return __factory()


async def get_session() -> AsyncIterator[AsyncSession]:
    async with create_session() as session:
        yield session
