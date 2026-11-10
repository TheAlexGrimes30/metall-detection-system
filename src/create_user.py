"""Provision a user: python -m src.create_user USERNAME."""

import argparse
import asyncio
import os
from getpass import getpass

from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from src.auth import password_hash
from src.db.db_session import close_db, create_session, global_init
from src.db.models.user import User


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('username')
    args = parser.parse_args()
    if not args.username.strip():
        parser.error('Username cannot be empty')
    password = getpass('Password: ')
    if len(password) < 8:
        parser.error('Password must contain at least 8 characters')
    if password != getpass('Confirm password: '):
        parser.error('Passwords do not match')
    await global_init(os.getenv('DATABASE_URL', ''))
    try:
        async with create_session() as session:
            if await session.scalar(select(User).where(User.name == args.username)):
                parser.error('Username already exists')
            session.add(User(name=args.username, password_hash=await run_in_threadpool(password_hash.hash, password)))
            await session.commit()
    finally:
        await close_db()
    print('User created')


if __name__ == '__main__':
    asyncio.run(main())
