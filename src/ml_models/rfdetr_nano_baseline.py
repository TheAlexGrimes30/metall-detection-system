from pathlib import Path

import supervision as sv
from PIL import Image

from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES


IMAGE_PATH = "images/test_photo_cat.jpg"
OUTPUT_PATH = "outputs/rfdetr/test_photo_cat.jpg"
WEIGHTS_PATH = "models/rf-detr-nano.pth"

Path("outputs/rfdetr").mkdir(parents=True, exist_ok=True)

model = RFDETRNano(
    pretrain_weights=WEIGHTS_PATH
)

image = Image.open(IMAGE_PATH).convert("RGB")

detections = model.predict(
    image,
    threshold=0.01
)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]

annotated = sv.BoxAnnotator(
    thickness=2
).annotate(
    scene=image.copy(),
    detections=detections
)

annotated = sv.LabelAnnotator().annotate(
    scene=annotated,
    detections=detections,
    labels=labels
)

annotated.save(OUTPUT_PATH)

for class_id, confidence in zip(
    detections.class_id,
    detections.confidence
):
    print(f"Class: {COCO_CLASSES[class_id]}")
    print(f"Confidence: {confidence:.4f}")

print(f"Saved to: {OUTPUT_PATH}")
