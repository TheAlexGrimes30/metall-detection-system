from datetime import datetime, UTC

from sqlalchemy import Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from ..db_session import DBBase


class DefectType(DBBase):
    __tablename__ = 'defect_type'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    dt_created: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    # backref relations
    # defects: 1:M relation from Defect
