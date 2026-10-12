from typing import TYPE_CHECKING

from datetime import datetime, UTC

from sqlalchemy import Integer, String, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column, WriteOnlyMapped, relationship

from ..db_session import DBBase

if TYPE_CHECKING:
    from .defect import Defect


class DefectType(DBBase):
    __tablename__ = 'defect_type'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    defects: WriteOnlyMapped['Defect'] = relationship('Defect', back_populates='defect_type')
    dt_created: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
