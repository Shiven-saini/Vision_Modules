# Robora Vision Modules

Robora Vision Modules is a Python library that provides plug and play vision skills reusable across Robora demos and inside robot “brains.”

This repository is part of Robora’s initiative to collaborate with students and researchers from different universities. The goal is to expand the reach of robotics combined with blockchain and create an open environment where knowledge, tools, and real world applications can grow through collaboration.

If you are a student, researcher, or developer, you are welcome to contribute. Fork the repository, make improvements, and submit a pull request. Together we can advance robotics x blockchain and push forward the adoption of physical AI.

---

## ✨ Features
- **Object Detection** (YOLOv8 wrapper)  
- **Image Segmentation** (SAM-lite or similar lightweight segmenter)  
- **Marker / Barcode Detection** (ArUco or QR)  
- **Unified Python API** for simple usage  
- **Command-line tools (CLI)** for quick demos  
- **Tiny evaluation script** for COCO-style datasets  

---

## 📂 Repository Structure
```
Vision_Modules/
├── README.md
├── requirements.txt
├── rvm/
│   ├── __init__.py
│   ├── api.py                # unified high-level API
│   ├── core/
│   │   ├── types.py          # dataclasses for boxes, masks, markers
│   │   └── visualize.py      # drawing utilities
│   ├── detect/
│   │   └── yolo.py           # detection wrapper
│   ├── segment/
│   │   └── sam_lite.py       # segmentation wrapper
│   ├── markers/
│   │   └── aruco.py          # marker detection
    |   └── barcodes.py       # QR Codes & Bar Codes detection
│   └── io/
│       ├── loader.py         # image, video, webcam loading
│       └── writer.py         # save JSON + annotated media
├── demos/
│   ├── detect_webcam.py
│   ├── detect_video.py
│   ├── segment_image.py
│   └── markers_image.py
├── eval/
│   └── coco_eval.py          # detection metrics + report.html
├── tests/
│   ├── test_api_smoke.py
│   ├── test_visualize.py
│   └── test_coco_eval.py
├── samples/
│   ├── shelf.jpg
│   └── tags.png
└── .github/
    └── workflows/
        └── ci.yml            # CI pipeline: run tests on push
```

---

## 🚀 Installation

We would recommend that you install the project inside a virtual environment to avoid dependency conflicts.

#### 1. Clone the repository
```bash
git clone https://github.com/RoboraDev/Vision_Modules
cd Vision_Modules
```
#### 2. Create and activate a virtual environment
##### Create virtual environment
```bash
python3.11 -m venv venv_rvm
```

##### Activate (Linux/Mac)
```bash
source venv_rvm/bin/activate
```

##### Activate (Windows)
```bash
venv_rvm\Scripts\activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Install the package in editable mode
```bash
pip install -e .
```

#### 🔥 Quick Install (alternative)

If you already have the required dependencies installed, you can skip steps 2–3 and install directly:
```bash
pip install -e .
```
---

## 📦 Requirements
- torch >= 2.2  
- ultralytics >= 8.1  
- opencv-python >= 4.9  
- numpy >= 1.26  
- matplotlib >= 3.8  
- pyzbar >= 0.1.9 (or `opencv-contrib-python` if using ArUco)  
- pycocotools >= 2.0.7  

---

## 🧑‍💻 Usage
### CLI Commands
```bash
rvm-detect --source path_or_webcam --model yolov8n.pt --out results/
rvm-segment --source images_dir --out results/
rvm-markers --source images_dir --out results/
rvm-eval-coco --images images_dir --ann annotations.json --out reports/
```

### Python API
You can also use **Vision Modules** directly in Python without the CLI.

#### 1. Object Detection
```python
from rvm.api import detect

results = detect(
    source="path/to/images_or_video",   # file, folder, or webcam index
    model="yolov8n.pt",                 # YOLO model checkpoint
    out_dir="results/"                  # output directory
)
print(results)
```

#### 2. Segmentation
```python
from rvm.api import segment

masks = segment(
    iamge_ơath="path/to/images_dir",
    out_dir="results/"
)
print(masks[0].shape)
```

#### 3. Markers
```python
from rvm.api import markers

output = markers(
    image_path="path/to/images_dir",
    out_dir="results/"
)
print(output)
```

#### 4. COCO Evaluation
```python
from rvm.api import coco_eval

metrics = coco_eval(
    pred_file="preds.json",          # predictions in COCO format
    ann_file="annotations.json",     # ground-truth annotations
    out_dir="reports/"
)
print(metrics)
```


---

## 🎥 Demos
We provide simple demo scripts for quick testing:

- `demos/detect_webcam.py` → run YOLO detection live from webcam  
- `demos/detect_video.py`  → detect objects in video, save annotated MP4 + JSON  
- `demos/segment_image.py` → run SAM-lite segmentation on an image  
- `demos/markers_image.py` → detect QR/ArUco markers in image  

Example:
```bash
python demos/detect_webcam.py --model yolov8n.pt
```

---

## 📊 Evaluation
Run COCO-style evaluation on predictions:
```bash
rvm-eval-coco --images path/to/images_dir --ann annotations.json --out reports/
```

This will output:
- Precision (AP@[0.5:0.95])
- Recall (AR@100)
- report.html (human-readable report)
- pr_curve.png (precision–recall curve)

---

## ✅ Tests & CI
We use pytest for testing and GitHub Actions for continuous integration.
Run all tests locally:
```bash
pytest -v
```
Tests include:
- Unit tests for each API function
- Integration tests for visualization
- Evaluation tests with minimal COCO-format data
  
CI automatically runs these tests on every pull request.

## 📌 Roadmap
- Updating  

---

## 🤝 Collaboration  

This repository is built with collaboration in mind. Robora is working closely with students, universities, and research groups to advance robotics and blockchain together.  

### How to contribute  
1. Fork this repository  
2. Create a new branch for your feature or fix  
3. Commit your changes  
4. Push your branch  
5. Open a pull request  

All contributions are welcome, whether through research ideas, code improvements, documentation, or new demos.  

---

🌐 Community and Links

[Website](https://robora.xyz)

[X](https://x.com/userobora)

[Telegram](https://t.me/roboratg)

[Medium](https://robora.medium.com)

---

## 👥 Authors
Maintained by **Robora**.  
Contributor: [@ncquy](https://github.com/ncquy), [@TianleiZhou](https://github.com/TianleiZhou), [@Shiven-saini](https://github.com/shiven-saini), *Updating...* 
