from pathlib import Path

import torch
from PIL import Image, ImageDraw

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)


IMAGE_PATH = "images/test_photo_cat.jpg"
OUTPUT_PATH = "outputs/faster_rcnn/test_photo_cat.jpg"

CONF_THRESHOLD = 0.25


# Create output directory
Path("outputs/faster_rcnn").mkdir(
    parents=True,
    exist_ok=True
)


# -------------------------
# 1. Load pretrained model
# -------------------------

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(
    weights=weights
)

model.eval()



class_names = weights.meta["categories"]


image = Image.open(
    IMAGE_PATH
).convert("RGB")



preprocess = weights.transforms()

image_tensor = preprocess(image)



with torch.no_grad():
    prediction = model(
        [image_tensor]
    )[0]


annotated = image.copy()

draw = ImageDraw.Draw(annotated)


for box, label, score in zip(
    prediction["boxes"],
    prediction["labels"],
    prediction["scores"]
):

    confidence = float(score)

    if confidence < CONF_THRESHOLD:
        continue

    class_id = int(label)

    class_name = class_names[class_id]

    x1, y1, x2, y2 = box.tolist()

    draw.rectangle(
        [x1, y1, x2, y2],
        outline="red",
        width=3
    )


    text = f"{class_name} {confidence:.2f}"

    draw.text(
        (x1, y1),
        text,
        fill="red"
    )


    print(f"Class: {class_name}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Box: {[x1, y1, x2, y2]}")
    print("-" * 30)

annotated.save(
    OUTPUT_PATH
)

print(f"Saved to: {OUTPUT_PATH}")