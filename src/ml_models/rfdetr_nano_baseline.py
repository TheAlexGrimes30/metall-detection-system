from pathlib import Path

import supervision as sv
from PIL import Image

from rfdetr import RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES


IMAGE_PATH = "images/test_photo_cat.jpg"
OUTPUT_PATH = "outputs/rfdetr/test_photo_cat.jpg"
WEIGHTS_PATH = "models/rfdetr_nano.pth"

Path("outputs/rfdetr").mkdir(parents=True, exist_ok=True)


model = RFDETRNano(
    # pretrain_weights="./rfdetr_nano.pth"
)

image = Image.open(IMAGE_PATH).convert("RGB")

detections = model.predict(
    image,
    threshold=0.05
)

print(detections)
print("Number of detections:", len(detections))


labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(
        detections.class_id,
        detections.confidence
    )
]

for class_id, confidence, box in zip(
    detections.class_id,
    detections.confidence,
    detections.xyxy
):
    print(
        f"Class: {COCO_CLASSES[class_id]}, "
        f"Confidence: {confidence:.4f}, "
        f"Box: {box}"
    )


annotated = image.copy()

annotated = sv.BoxAnnotator(
    thickness=3
).annotate(
    annotated,
    detections
)

annotated = sv.LabelAnnotator().annotate(
    annotated,
    detections,
    labels
)

annotated.save(OUTPUT_PATH)

print(f"Saved to: {OUTPUT_PATH}")
