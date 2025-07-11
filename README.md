# 🦴 AI-Powered X-Ray Fracture Detection

This project is a deep learning-based medical imaging tool designed to identify wrist fractures in X-ray images using a YOLOv8 classification model. Built for deployment on the Jetson Orin Nano, it enables fast, on-device analysis of medical scans with automatic triage logic based on confidence levels — helping reduce wait times and assist clinical decision-making.

## 🔍 Features

- 📸 Classifies X-ray images as **fractured** or **normal**
- ⚙️ Built using **YOLOv8** and the **Ultralytics** API
- 🧠 Confidence-based triage logic to recommend action
- 🐧 Designed to run on Jetson Linux devices (e.g. Orin Nano)
- 📂 Scans any image placed in a target folder or selected individually
- 📝 Clear terminal outputs for file name, prediction, confidence, and decision

---

## 🚀 Example Output
📸 Image Path:     /home/nvidia07/X-RAI/toscan/example.jpg   
🧠 Prediction:     fractured   
📈 Confidence:     88.30%   
✅ Decision:       Fracture detected with high confidence.   

---

## 📁 Project Structure

X-RAI   
├── runs/                   
│   └── classify/train/weights/best.pt   
├── toscan/             
│   └── example.jpg   
├── main.py                   
├── README.md              

---

## ⚙️ Requirements

- Python 3.8+
- Ultralytics YOLOv8
- Pillow (PIL)
- Jetson Orin Nano (recommended, but not required)

---

## 🧪 How to Use

🔹 Classify a Single Image

1. Place your image inside the `toscan/` folder.
2. Open `main.py` and update this line:

   IMAGE_FILENAME = "your_image_name.jpg"

3. Run the script:

   python3 main.py

---

## 🔣 Sample Script

from ultralytics import YOLO   
from PIL import Image   
import os   

MODEL_PATH = "/home/nvidia07/X-RAI/runs/classify/train/weights/best.pt"   
TOSCAN_DIR = "/home/nvidia07/X-RAI/toscan"   
IMAGE_FILENAME = "INPUT_IMAGE_HERE.jpg"   
IMAGE_PATH = os.path.join(TOSCAN_DIR, IMAGE_FILENAME)   

model = YOLO(MODEL_PATH)   
results = model.predict(source=IMAGE_PATH, save=False)   

probs = results[0].probs   
class_names = model.names   
pred_index = probs.top1   
pred_label = class_names[pred_index]   
confidence = probs.data[pred_index].item() * 100   

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

print("---------------------------------------------------")   
print(f"📸 Image Path:     {IMAGE_PATH}")   
print(f"🧠 Prediction:     {pred_label}")   
print(f"📈 Confidence:     {confidence:.2f}%")   
print(f"✅ Decision:       {make_decision(pred_label, confidence)}")   
print("---------------------------------------------------")

---

🎥 Demo Video

Watch a full walkthrough of the project, including how it works and what it looks like in action:

👉 https://youtu.be/Y95pP8V3GVQ

---

## 🔄 Future Improvements

- [ ] Web interface or GUI for easier use
- [ ] Integration with DICOM viewers
- [ ] Heatmap visualization for attention areas
- [ ] Automated batch scanning mode

---

## 🙋‍♂️ Author

Lachlan – Medical AI Developer
