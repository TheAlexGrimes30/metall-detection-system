from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db_session import DBBase

if TYPE_CHECKING:
    from .defect_type import DefectType
    from .inspection import Inspection


class Defect(DBBase):
    __tablename__ = 'defect'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    type_id: Mapped[int] = mapped_column(ForeignKey('defect_type.id'))
    type: Mapped['DefectType'] = relationship(
        'DefectType', foreign_keys=type_id, back_populates='defects'
    )
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspection.id'))
    inspection: Mapped['Inspection'] = relationship(
        'Inspection', foreign_keys=inspection_id, back_populates='defects'
    )
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
