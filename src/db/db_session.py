from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

DBBase = declarative_base()
__factory = None


def global_init(url: str):
    url = url.strip().replace('postgres', 'postgresql')
    global engine, __factory
    if __factory:
        return
    if not url:
        raise Exception('Некорректный адрес БД')
    engine = create_engine(url, echo=False, poolclass=NullPool)
    __factory = sessionmaker(bind=engine)
    from . import __all_models # type: ignore
    DBBase.metadata.create_all(engine)


def create_session() -> Optional[Session]:
    global __factory
    if __factory is not None:
        return __factory()