from datetime import datetime, UTC

from sqlalchemy import Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from ..db_session import DBBase


class User(DBBase):
    __tablename__ = 'user'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    dt_created: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # backref relations
    # images: 1:M relation from Image
    # logs: 1:M relation from Log
