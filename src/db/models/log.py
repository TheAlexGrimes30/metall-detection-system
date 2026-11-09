from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db_session import DBBase

if TYPE_CHECKING:
    from .inspection import Inspection
    from .ml_model import MLModel
    from .user import User


class Log(DBBase):
    """
    Keeping ids of some entities with events that are not self-explanatory with their own fields (e.g., dt_created)
    """

    __tablename__ = 'log'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(ForeignKey('user.id'))
    user: Mapped['User'] = relationship(
        'User', foreign_keys=user_id, back_populates='logs'
    )

    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspection.id'))
    inspection: Mapped['Inspection'] = relationship(
        'Inspection', foreign_keys=inspection_id, back_populates='logs'
    )

    ml_model_id: Mapped[int] = mapped_column(ForeignKey('ml_model.id'))
    ml_model: Mapped['MLModel'] = relationship(
        'MLModel', foreign_keys=ml_model_id, back_populates='logs'
    )

    event: Mapped[str] = mapped_column(String, nullable=False)
    dt: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
