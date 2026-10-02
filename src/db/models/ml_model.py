from datetime import datetime, UTC

from sqlalchemy import Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from ..db_session import DBBase


class MLModel(DBBase):
    __tablename__ = 'ml_model'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    dt_created: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    # backref relations
    # ml_metrics: 1:M relation from MLMetrics
    # logs: 1:M relation from Logs
