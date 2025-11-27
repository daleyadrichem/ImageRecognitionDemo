# AI Vision Demo

A real-time computer vision demo showcasing **image classification**, **object detection**, **semantic segmentation**, and **neural style transfer**, built for workshops and educational presentations.

---

## 🚀 Features

### **1. Image Classification**
Predicts a single label for the entire frame using **ResNet-50 (ImageNet)**.

### **2. Object Detection**
Finds and labels multiple objects using **Faster R-CNN (COCO)**.

### **3. Semantic Segmentation**
Assigns a class label to every pixel using **DeepLabV3 (COCO/VOC)**.

### **4. Neural Style Transfer**
Transforms the webcam feed into the artistic style of a reference painting (e.g., Van Gogh, Picasso) using fast neural style transfer models.

---

## 🎛️ Controls

| Key | Action |
|-----|--------|
| **1** | Image Classification |
| **2** | Object Detection |
| **3** | Semantic Segmentation |
| **4** | Style Transfer |
| **q** | Quit |

---

## 📦 Installation (uv)

This project uses **uv** for environment and dependency management.
```bash
uv sync
```
This automatically creates the .venv environment and installs all dependencies listed in pyproject.toml.

## ▶️ Running the Demo

```bash
uv run main.py
```
A webcam window will open. Switch between the four AI demos using the number keys.

## 🧱 Project Structure

```bash
ai-vision-demo/
├── classification_demo.py   # Image classification (ResNet-50)
├── detection_demo.py        # Object detection (Faster R-CNN)
├── segmentation_demo.py     # Semantic segmentation (DeepLabV3)
├── style_transfer_demo.py   # Neural style transfer
├── utils.py                 # Shared utilities (drawing, FPS, device)
├── main.py                  # Application entry point
└── pyproject.toml           # Dependencies (uv)
```

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch
- TorchVision
- OpenCV
- Pillow
- NumPy
- uv (dependency & environment manager)

## 📚 Purpose

This repository is designed to help explain:

- How neural networks process visual information
- Differences between classification, detection, and segmentation
- How style transfer uses a reference painting
- The progression from simple to complex computer vision tasks

It is optimized for live demos, workshops, and teaching.

## 📄 License

MIT License — feel free to adapt this demo for workshops or training sessions.

```yaml

---

If you want, I can also produce:

- A version with **example images/GIFs**  
- A version with **badges** (Python version, license, etc.)  
- A more **technical README** targeting developers instead of workshop audiences

```