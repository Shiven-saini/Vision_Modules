# Robora Vision Modules

Robora Vision Modules

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
│   │   └── aruco.py          # marker / QR detection
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

```bash
Updating
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

### Python API
```python
Updating
```

### CLI Commands
```bash
rvm-detect --source path_or_webcam --model yolov8n.pt --out results/
rvm-segment --source images_dir --out results/
rvm-markers --source images_dir --out results/
rvm-eval-coco --images images_dir --ann annotations.json --out reports/
```

---

## 🎥 Demos
- `demos/detect_webcam.py` → run YOLO detection live from webcam  
- `demos/detect_video.py` → detect objects in video, save annotated MP4 + JSON  
- `demos/segment_image.py` → run SAM-lite segmentation on an image  
- `demos/markers_image.py` → detect QR/ArUco markers in image  

---

## 📊 Evaluation
Run COCO-style evaluation on a small dataset:

```bash
Updating
```

Outputs **precision, recall, and report.html**.

---

## ✅ Tests & CI
- Updating

---

## 📌 Roadmap
- Updating  

---

🤝 Collaboration
This repository is built with collaboration in mind. Robora is working closely with students, universities, and research groups to advance robotics and blockchain together.

How to contribute

Fork this repository

Create a new branch for your feature or fix

Commit your changes

Push your branch

Open a pull request

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
Contributor: [@ncquy](https://github.com/ncquy), [@TianleiZhou](https://github.com/TianleiZhou), *Updating...* 
