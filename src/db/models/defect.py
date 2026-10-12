from sqlalchemy import Integer, Float, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from ..db_session import DBBase


class Defect(DBBase):
    __tablename__ = 'defect'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type_id: Mapped[int] = mapped_column(ForeignKey('defect_type.id'))
    type = relationship(
        'DefectType',
        foreign_keys=type_id,
        backref=backref('defects', lazy='dynamic')
    )
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspection.id'))
    inspection = relationship(
        'Inspection',
        foreign_keys=inspection_id,
        backref=backref('defects', lazy='dynamic')
    )
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
