from datetime import datetime, UTC

from sqlalchemy import Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship, backref

from ..db_session import DBBase


class Log(DBBase):
    '''
    Keeping ids of some entities with events that are not self-explanatory with their own fields (e.g., dt_created)
    '''

    __tablename__ = 'log'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    user_id: Mapped[int] = mapped_column(ForeignKey('user.id'))
    user = relationship(
        'User',
        foreign_keys=user_id,
        backref=backref('logs', lazy='dynamic')
    )
    
    inspection_id: Mapped[int] = mapped_column(ForeignKey('inspection.id'))
    inspection = relationship(
        'Inspection',
        foreign_keys=inspection_id,
        backref=backref('logs', lazy='dynamic')
    )

    ml_model_id: Mapped[int] = mapped_column(ForeignKey('ml_model.id'))
    ml_model = relationship(
        'MLModel',
        foreign_keys=ml_model_id,
        backref=backref('logs', lazy='dynamic')
    )

    event: Mapped[str] = mapped_column(String, nullable=False)
    dt: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
