from datetime import datetime, UTC

from sqlalchemy import Integer, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from ..db_session import DBBase


class Inspection(DBBase):
    __tablename__ = 'inspection'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_in_id: Mapped[int] = mapped_column(ForeignKey('image.id'))
    image_in = relationship(
        'Image',
        foreign_keys=image_in_id,
        backref=backref('inspections_in', lazy='dynamic')
    )
    image_out_id: Mapped[int] = mapped_column(ForeignKey('image.id'), nullable=True)
    image_out = relationship(
        'Image',
        foreign_keys=image_out_id,
        backref=backref('inspections_out', lazy='dynamic')
    )
    ml_model_id: Mapped[int] = mapped_column(ForeignKey('ml_model.id'))
    ml_model = relationship(
        'MLModel',
        foriegn_keys=ml_model_id,
        backref=backref('inspections', lazy='dynamic')
    )
    dt_started: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    dt_finished: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # backref relations
    # defects: 1:M relation from Defect
    # logs: 1:M relation from Log
