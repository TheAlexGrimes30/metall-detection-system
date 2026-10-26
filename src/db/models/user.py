from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, WriteOnlyMapped, mapped_column, relationship

from ..db_session import DBBase

if TYPE_CHECKING:
    from .image import Image
    from .log import Log


class User(DBBase):
    __tablename__ = 'user'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    password_hash: Mapped[str] = mapped_column(String, nullable=False)
    images: WriteOnlyMapped['Image'] = relationship('Image', back_populates='user')
    logs: WriteOnlyMapped['Log'] = relationship('Log', back_populates='user')
    dt_created: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
