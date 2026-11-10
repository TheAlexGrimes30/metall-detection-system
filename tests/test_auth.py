from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import jwt
import pytest

from main import app
from src import auth
from src.db.db_session import get_session
from src.db.models.user import User

pytestmark = pytest.mark.anyio


@pytest.fixture
def user():
    return User(id=7, name='alice', password_hash=auth.password_hash.hash('correct-password'), is_deleted=False)


@pytest.fixture
def session(user):
    return AsyncMock(scalar=AsyncMock(return_value=user), get=AsyncMock(return_value=user))


@pytest.fixture
async def client(session):
    async def override_session():
        yield session

    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_session] = override_session
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


def assert_unauthorized(response):
    assert response.status_code == 401
    assert response.headers['www-authenticate'] == 'Bearer'


async def test_login_and_protected_routes(client, user, session):
    response = await client.post('/auth/token', data={'username': user.name, 'password': 'correct-password'})
    assert response.status_code == 200
    assert response.json()['token_type'] == 'bearer'
    token = response.json()['access_token']
    payload = jwt.decode(token, auth.get_secret_key(), algorithms=['HS256'])
    assert payload['sub'] == str(user.id)
    assert payload['exp'] - payload['iat'] == 1800
    headers = {'Authorization': f'Bearer {token}'}
    response = await client.get('/auth/me', headers=headers)
    assert response.status_code == 200
    assert response.json() == {'id': user.id, 'name': user.name}
    session.get.assert_awaited_with(User, user.id)
    response = await client.post('/initialize', headers=headers)
    assert response.status_code == 200
    assert response.json() == {'message': 'Initialized'}


@pytest.mark.parametrize('case', ['wrong_password', 'unknown_user', 'deleted_user'])
async def test_login_rejects_invalid_credentials(client, session, user, case):
    if case == 'unknown_user':
        session.scalar.return_value = None
    if case == 'deleted_user':
        user.is_deleted = True
    password = 'wrong' if case == 'wrong_password' else 'correct-password'
    response = await client.post('/auth/token', data={'username': 'alice', 'password': password})
    assert_unauthorized(response)
    assert response.json() == {'detail': 'Invalid credentials'}


async def test_login_requires_form_fields(client):
    response = await client.post('/auth/token', data={'username': 'alice'})
    assert response.status_code == 422


@pytest.mark.parametrize('path,method', [('/auth/me', 'get'), ('/initialize', 'post')])
@pytest.mark.parametrize('authorization', [None, 'Basic abc', 'Bearer broken-token'])
async def test_protected_routes_reject_invalid_authorization(client, path, method, authorization):
    headers = {} if authorization is None else {'Authorization': authorization}
    assert_unauthorized(await client.request(method, path, headers=headers))


@pytest.mark.parametrize('case', ['expired', 'wrong_signature', 'wrong_algorithm', 'missing_sub', 'missing_exp', 'missing_iat', 'invalid_sub', 'future_iat'])
async def test_rejects_invalid_jwt(client, session, case):
    now = datetime.now(UTC)
    payload = {'sub': '7', 'iat': now, 'exp': now + timedelta(minutes=30)}
    key = auth.get_secret_key()
    algorithm = 'HS256'
    if case == 'expired':
        payload['exp'] = now - timedelta(seconds=1)
    elif case == 'wrong_signature':
        key = 'a-different-secret-key-at-least-32-characters'
    elif case == 'wrong_algorithm':
        algorithm = 'HS384'
    elif case.startswith('missing_'):
        del payload[case.removeprefix('missing_')]
    elif case == 'invalid_sub':
        payload['sub'] = 'not-an-integer'
    elif case == 'future_iat':
        payload['iat'] = now + timedelta(hours=1)
    token = jwt.encode(payload, key, algorithm=algorithm)
    assert_unauthorized(await client.get('/auth/me', headers={'Authorization': f'Bearer {token}'}))
    session.get.assert_not_awaited()


@pytest.mark.parametrize('deleted', [False, True])
async def test_token_rejected_after_user_removed(client, session, user, deleted):
    token = auth.create_access_token(user.id)
    if deleted:
        user.is_deleted = True
    else:
        session.get.return_value = None
    assert_unauthorized(await client.get('/auth/me', headers={'Authorization': f'Bearer {token}'}))


@pytest.mark.parametrize('secret', ['', 'short', 'a' * 31])
async def test_rejects_unsafe_secret(monkeypatch, secret):
    monkeypatch.setenv('JWT_SECRET_KEY', secret)
    with pytest.raises(RuntimeError, match='JWT_SECRET_KEY'):
        auth.create_access_token(7)


async def test_password_hash_is_salted():
    first = auth.password_hash.hash('password')
    second = auth.password_hash.hash('password')
    assert first != second
    assert auth.password_hash.verify('password', first)
    assert not auth.password_hash.verify('wrong', first)
