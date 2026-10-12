from typing import TYPE_CHECKING

from datetime import datetime, UTC

from sqlalchemy import Integer, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship, WriteOnlyMapped

from ..db_session import DBBase

if TYPE_CHECKING:
    from .image import Image
    from .ml_model import MLModel
    from .defect import Defect
    from .log import Log


class Inspection(DBBase):
    __tablename__ = 'inspection'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_in_id: Mapped[int] = mapped_column(ForeignKey('image.id'))
    image_in: WriteOnlyMapped['Image'] = relationship(
        'Image',
        foreign_keys=image_in_id,
        back_populates='inspections_in'
    )
    image_out_id: Mapped[int] = mapped_column(ForeignKey('image.id'), nullable=True)
    image_out: WriteOnlyMapped['Image'] = relationship(
        'Image',
        foreign_keys=image_out_id,
        back_populates='inspections_out'
    )
    ml_model_id: Mapped[int] = mapped_column(ForeignKey('ml_model.id'))
    ml_model: WriteOnlyMapped['MLModel'] = relationship(
        'MLModel',
        foreign_keys=ml_model_id,
        back_populates='inspections'
    )
    defects: WriteOnlyMapped['Defect'] = relationship('Defect', back_populates='inspection')
    logs: WriteOnlyMapped['Log'] = relationship('Log', back_populates='inspection')
    dt_started: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    dt_finished: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
