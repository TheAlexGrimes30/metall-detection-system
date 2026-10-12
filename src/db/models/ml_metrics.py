from typing import TYPE_CHECKING

from datetime import datetime, UTC

from sqlalchemy import Integer, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, WriteOnlyMapped

from ..db_session import DBBase

if TYPE_CHECKING:
    from .ml_model import MLModel


class MLMetrics(DBBase):
    __tablename__ = 'ml_metrics'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ml_model_id: Mapped[int] = mapped_column(ForeignKey('ml_model.id'))
    ml_model: WriteOnlyMapped['MLModel'] = relationship(
        'MLModel',
        foreign_keys=ml_model_id,
        back_populates='metrics'
    )
    iou: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    dt_created: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
