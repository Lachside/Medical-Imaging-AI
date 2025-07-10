from ultralytics import YOLO
from PIL import Image
import os

# 🔧 REQUIRED: Path to your trained model (.pt file)
MODEL_PATH = "/home/nvidia07/X-RAI/runs/classify/train/weights/best.pt"

# 🔧 Folder containing images to scan
TOSCAN_DIR = "/home/nvidia07/X-RAI/toscan"

# 👇 Only type the image filename here
IMAGE_FILENAME = "EExtremelyHard2.jpg"  # ⬅️ change this line only

# 🔧 Automatically create full image path
IMAGE_PATH = os.path.join(TOSCAN_DIR, IMAGE_FILENAME)

# === Load the YOLOv8 classification model ===
model = YOLO(MODEL_PATH)

# === Run prediction ===
results = model.predict(source=IMAGE_PATH, save=False)

# === Get prediction info ===
probs = results[0].probs
class_names = model.names
pred_index = probs.top1
pred_label = class_names[pred_index]
confidence = probs.data[pred_index].item() * 100

# === Improved Decision Logic ===
def make_decision(label, conf):
    if "fracture" in label.lower():
        if conf >= 70:
            return "Fracture detected with high confidence."
        elif 50 <= conf < 70:
            return "Possible fracture detected – refer to doctor."
        else:
            return "Possible fracture, but AI is unsure – refer to doctor."
    else:
        if conf >= 70:
            return "No fracture detected with high confidence."
        else:
            return "No fracture detected, but AI is unsure – refer to doctor."

# === Print results ===
print("---------------------------------------------------")
print(f"📸 Image Path:     {IMAGE_PATH}")
print(f"🧠 Prediction:     {pred_label}")
print(f"📈 Confidence:     {confidence:.2f}%")
print(f"✅ Decision:       {make_decision(pred_label, confidence)}")
print("---------------------------------------------------")
