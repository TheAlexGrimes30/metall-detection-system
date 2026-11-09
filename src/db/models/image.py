import enum
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, WriteOnlyMapped, mapped_column, relationship

from ..db_session import DBBase

if TYPE_CHECKING:
    from .inspection import Inspection
    from .user import User


class ImageType(enum.Enum):
    ORIGINAL = 1
    PREPROCESSED = 2
    RESULT = 3


class Image(DBBase):
    __tablename__ = 'image'

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[int] = mapped_column(ForeignKey('user.id'))
    user: Mapped['User'] = relationship(
        'User', foreign_keys=user_id, back_populates='images'
    )
    type: Mapped[ImageType] = mapped_column(Enum(ImageType), default=ImageType.ORIGINAL)
    original_image_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('image.id'))
    original_image: Mapped['Image'] = relationship(
        'Image', foreign_keys=original_image_id, back_populates='derivative_images'
    )
    derivative_images: WriteOnlyMapped['Image'] = relationship(
        'Image', back_populates='original_image'
    )
    inspections_in: WriteOnlyMapped['Inspection'] = relationship(
        'Inspection', back_populates='image_in'
    )
    inspections_out: WriteOnlyMapped['Inspection'] = relationship(
        'Inspection', back_populates='image_out'
    )
    metadata_: Mapped[str] = mapped_column(String, nullable=True)
    dt_uploaded: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
