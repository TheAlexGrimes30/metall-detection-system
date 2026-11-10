from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db import db_session as db


@pytest.mark.parametrize('scheme', ['postgres', 'postgresql', 'postgresql+asyncpg'])
def test_normalizes_postgresql_url(scheme):
    assert db.get_database_url(f'  {scheme}://user:p%40ss@localhost:5432/test  ') == (
        'postgresql+asyncpg://user:p%40ss@localhost:5432/test'
    )


@pytest.mark.parametrize('url', ['', '   ', 'sqlite:///test.db', 'mysql://user@localhost/test'])
def test_rejects_unsupported_database_url(url):
    with pytest.raises(ValueError):
        db.get_database_url(url)


@pytest.fixture
def isolated_db(monkeypatch):
    monkeypatch.setattr(db, 'engine', None)
    monkeypatch.setattr(db, '__factory', None)


def test_session_requires_initialization(isolated_db):
    with pytest.raises(RuntimeError, match='не инициализирована'):
        db.create_session()


@pytest.mark.anyio
async def test_initialization_and_shutdown(isolated_db, monkeypatch):
    connection = AsyncMock()
    engine = MagicMock()
    engine.begin.return_value.__aenter__ = AsyncMock(return_value=connection)
    engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
    engine.dispose = AsyncMock()
    create_engine = MagicMock(return_value=engine)
    factory = MagicMock()
    make_factory = MagicMock(return_value=factory)
    monkeypatch.setattr(db, 'create_async_engine', create_engine)
    monkeypatch.setattr(db, 'async_sessionmaker', make_factory)

    await db.global_init('postgres://user@localhost/test')
    create_engine.assert_called_once_with('postgresql+asyncpg://user@localhost/test', echo=False)
    connection.run_sync.assert_awaited_once_with(db.DBBase.metadata.create_all)
    make_factory.assert_called_once_with(engine, expire_on_commit=False)
    assert db.create_session() is factory.return_value
    await db.global_init('postgres://user@localhost/test')
    create_engine.assert_called_once()
    await db.close_db()
    engine.dispose.assert_awaited_once()
    assert db.engine is None
    with pytest.raises(RuntimeError):
        db.create_session()
    await db.close_db()


@pytest.mark.anyio
async def test_failed_initialization_disposes_engine(isolated_db, monkeypatch):
    engine = MagicMock()
    engine.begin.return_value.__aenter__ = AsyncMock(side_effect=OSError('connection failed'))
    engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)
    engine.dispose = AsyncMock()
    monkeypatch.setattr(db, 'create_async_engine', MagicMock(return_value=engine))
    with pytest.raises(OSError, match='connection failed'):
        await db.global_init('postgresql://user@localhost/test')
    engine.dispose.assert_awaited_once()
    assert db.engine is None
    with pytest.raises(RuntimeError):
        db.create_session()


@pytest.mark.anyio
async def test_session_dependency_closes_session(monkeypatch):
    session = AsyncMock()
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(db, 'create_session', MagicMock(return_value=context))
    dependency = db.get_session()
    assert await anext(dependency) is session
    await dependency.aclose()
    context.__aexit__.assert_awaited_once()
