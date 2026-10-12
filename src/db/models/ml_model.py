from datetime import datetime, UTC

from sqlalchemy import Integer, String, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from ..db_session import DBBase


class MLModel(DBBase):
    __tablename__ = 'ml_model'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=True)
    valid_from: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    # datetime.min = 01-01-01 00:00
    # if valid_to == datetime.min then it is the active version
    valid_to: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.min)
    dt_created: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # backref relations
    # ml_metrics: 1:M relation from MLMetrics
    # logs: 1:M relation from Logs
