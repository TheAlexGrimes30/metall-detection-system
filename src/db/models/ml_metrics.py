from datetime import datetime, UTC

from sqlalchemy import Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db_session import DBBase


class MLMetrics(DBBase):
    __tablename__ = 'ml_metrics'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ml_model_id: Mapped[int] = mapped_column(ForeignKey('ml_model.id'))
    ml_model = relationship('MLModel', foreign_keys=ml_model_id, backref='metrics')
    ml_model_version: Mapped[int] = mapped_column(Integer, nullable=False)
    iou: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    dt_created: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
