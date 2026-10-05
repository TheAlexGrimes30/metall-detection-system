from ultralytics import YOLO
from pathlib import Path

IMAGE_PATH = "images/test_photo_cat.jpg"
OUTPUT_PATH = "outputs/yolo/test_photo_cat.jpg"

Path("outputs/yolo").mkdir(parents=True, exist_ok=True)

model = YOLO("models/yolo11n.pt")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=False
)

for result in results:

    result.save(filename=OUTPUT_PATH)

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = result.names[cls_id]

        print(f"Class: {class_name}")
        print(f"Confidence: {conf:.4f}")

print(f"Saved to: {OUTPUT_PATH}")
