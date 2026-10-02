from .models.user import User
from .models.inspection import Inspection
from .models.image import Image, ImageType
from .models.defect_type import DefectType
from .models.defect import Defect
from .models.ml_model import MLModel
from .models.ml_metrics import MLMetrics
from .models.log import Log

# TODO: use lazy='dynamic' in all backrefs

__all__ = [
    'User', 'Inspection', 'Image', 'ImageType',
    'DefectType', 'Defect', 'MLModel', 'MLMetrics', 'Log'
]
