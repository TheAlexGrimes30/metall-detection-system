from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, WriteOnlyMapped, mapped_column, relationship

from ..db_session import DBBase

if TYPE_CHECKING:
    from .defect import Defect


class DefectType(DBBase):
    __tablename__ = 'defect_type'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    defects: WriteOnlyMapped['Defect'] = relationship(
        'Defect', back_populates='type'
    )
    dt_created: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
