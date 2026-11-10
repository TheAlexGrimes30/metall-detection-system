import pytest

from src.db import __all_models  # noqa: F401


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture(autouse=True)
def jwt_secret(monkeypatch):
    monkeypatch.setenv('JWT_SECRET_KEY', 'test-secret-key-for-authentication-only-12345678901234567890')
