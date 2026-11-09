import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, WriteOnlyMapped, mapped_column, relationship

from ..db_session import DBBase

if TYPE_CHECKING:
    from .defect import Defect
    from .image import Image
    from .log import Log
    from .ml_model import MLModel


class Inspection(DBBase):
    __tablename__ = 'inspection'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_in_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('image.id'))
    image_in: Mapped['Image'] = relationship(
        'Image', foreign_keys=image_in_id, back_populates='inspections_in'
    )
    image_out_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('image.id'), nullable=True)
    image_out: Mapped['Image'] = relationship(
        'Image', foreign_keys=image_out_id, back_populates='inspections_out'
    )
    ml_model_id: Mapped[int] = mapped_column(ForeignKey('ml_model.id'))
    ml_model: Mapped['MLModel'] = relationship(
        'MLModel', foreign_keys=ml_model_id, back_populates='inspections'
    )
    defects: WriteOnlyMapped['Defect'] = relationship(
        'Defect', back_populates='inspection'
    )
    logs: WriteOnlyMapped['Log'] = relationship('Log', back_populates='inspection')
    dt_started: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    dt_finished: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
