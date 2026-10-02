from datetime import datetime, UTC
import uuid
import enum

from sqlalchemy import String, DateTime, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db_session import DBBase


class ImageType(enum.Enum):
    ORIGINAL = 1
    PREPROCESSED = 2
    RESULT = 3


class Image(DBBase):
    __tablename__ = 'image'

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey('user.id'))
    user = relationship('User', foreign_keys=user_id, backref='images')
    type: Mapped[enum.Enum] = mapped_column(Enum(ImageType), default=ImageType.ORIGINAL)
    original_image_id: Mapped[int] = mapped_column(ForeignKey('image.id'))
    original_image = relationship('Image', foreign_keys=original_image_id, backref='derivative_images')
    metadata: Mapped[str] = mapped_column(String, nullable=True)
    dt_uploaded: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    # backref relations
    # derivative_images: 1:M relation from Image
    # inspections_in: 1:M relation from Inspection
    # inspections_out: 1:M relation from Inspection (1:1 actually)
