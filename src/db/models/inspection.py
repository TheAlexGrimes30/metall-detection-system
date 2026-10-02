from datetime import datetime, UTC

from sqlalchemy import Integer, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db_session import DBBase


class Inspection(DBBase):
    __tablename__ = 'inspection'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    image_in_id: Mapped[int] = mapped_column(ForeignKey('image.id'))
    image_in = relationship('Image', foreign_keys=image_in_id, backref='inspections_in')
    image_out_id: Mapped[int] = mapped_column(ForeignKey('image.id'), nullable=True)
    image_out = relationship('Image', foreign_keys=image_out_id, backref='inspections_out')
    dt_started: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    dt_finished: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # backref relations
    # defects: 1:M relation from Defect
    # logs: 1:M relation from Log
