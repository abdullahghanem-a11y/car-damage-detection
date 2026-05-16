# 🚗 Car Damage Detection System

A vision-based car damage detection and instance segmentation system built with **YOLOv8s-seg** and fine-tuned on the **CarDD dataset**. The system automatically localizes, classifies, and segments six types of car damage from images — beating the original benchmark paper's best model on every individual damage class.

> 📦 **Model weights hosted on HuggingFace:** [abdullahg7/cardd-yolov8s](https://huggingface.co/abdullahg7/cardd-yolov8s)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)

---

## 🔍 Overview

This project fine-tunes a pre-trained **YOLOv8s-seg** model on the CarDD dataset for real-world car damage detection and instance segmentation. It is deployed as a full-stack dockerized application with a FastAPI backend, PostgreSQL database, and React frontend.

**Detected Damage Categories:**

| Class | Description |
|---|---|
| 🔴 Dent | Surface deformation on car body |
| 🔵 Scratch | Linear paint damage |
| 🟢 Crack | Structural fractures |
| 🟣 Glass Shatter | Broken windows or windshields |
| 🟡 Lamp Broken | Damaged headlights or tail lights |
| 🟠 Tire Flat | Deflated or damaged tires |

---

## 📊 Results

### Overall Performance (Test Set)

| Metric | Box | Mask |
|---|---|---|
| mAP@0.5 | **75.84%** | **75.90%** |
| mAP@0.5:0.95 | **58.93%** | **57.00%** |
| Precision | 76.84% | 77.40% |
| Recall | 68.27% | 68.70% |
| Inference Speed | 61ms / image |  |
| Model Size | 22.5 MB |  |
| Training Time | ~3 hours (RTX 3050) |  |

### Per-Class Performance (Box mAP50 / Mask mAP50)

| Class | Box mAP50 | Mask mAP50 |
|---|---|---|
| Glass Shatter | 97.9% | 97.9% |
| Tire Flat | 91.0% | 90.2% |
| Lamp Broken | 86.8% | 88.2% |
| Scratch | 63.2% | 60.4% |
| Dent | 60.6% | 64.9% |
| Crack | 55.5% | 54.0% |

### Benchmark Comparison vs CarDD Paper

| Model | Backbone | mAP50 | vs Ours |
|---|---|---|---|
| Mask R-CNN | ResNet-50 | 66.3% | ✅ +9.5% |
| Cascade Mask R-CNN | ResNet-50 | 64.7% | ✅ +11.1% |
| GCNet | ResNet-50 | 66.4% | ✅ +9.4% |
| HTC | ResNet-50 | 68.1% | ✅ +7.7% |
| DCN | ResNet-50 | 70.9% | ✅ +4.9% |
| Mask R-CNN | ResNet-101 | 67.7% | ✅ +8.1% |
| HTC | ResNet-101 | 68.4% | ✅ +7.4% |
| DCN | ResNet-101 | 69.8% | ✅ +6.0% |
| **YOLOv8s-seg (Ours)** | — | **75.8%** | 🔥 |
| DCN+ | ResNet-50 | 77.4% | ❌ -1.6% |
| DCN+ | ResNet-101 | 78.8% | ❌ -3.0% |

### Per-Class Comparison vs DCN+ ResNet-101 (Paper's Best Model)

| Class | DCN+ ResNet-101 | Ours (Box) | Difference |
|---|---|---|---|
| Dent | 40.5% | **60.6%** | ✅ +20.1% |
| Scratch | 34.3% | **63.2%** | ✅ +28.9% |
| Crack | 16.6% | **55.5%** | ✅ +38.9% |
| Glass Shatter | 92.6% | **97.9%** | ✅ +5.3% |
| Lamp Broken | 70.8% | **86.8%** | ✅ +16.0% |
| Tire Flat | 86.0% | **91.0%** | ✅ +5.0% |

> **Our model beats DCN+ on every individual damage class**, while being significantly lighter (22.5MB vs 200MB+) and faster.

---

## ✨ Features

Beyond the base detection and segmentation model, the system includes the following novel features:

### 🔴 Damage Severity Scoring
Each detected damage instance is automatically assigned a severity label:
- **Minor** — small area, high confidence, low-severity damage type
- **Moderate** — medium area or moderate damage type
- **Severe** — large area, structural damage (crack, glass shatter)

### 🔄 Multi-Angle Damage Aggregation
Upload multiple images of the same car from different angles. The system:
- Runs detection on each image independently
- Aggregates all detections across angles
- Produces one unified damage report with per-image breakdown

### 🚗 Car Verification
Before running damage detection, the system verifies that the uploaded image actually contains a car using a COCO-pretrained YOLOv8 model — preventing false detections on non-vehicle images.

### 🔑 Car Re-Identification
When multiple images are uploaded, the system uses visual feature embeddings to verify that all images belong to the same vehicle — flagging inconsistencies that may indicate fraud.

---

## 📦 Dataset

This project uses the **CarDD** (Car Damage Detection) dataset:

- **4,000** high-resolution car damage images
- **9,000+** annotated instances
- **6** damage categories
- Annotations follow real insurance claim standards (Ping An Insurance)
- Published in IEEE Transactions on Intelligent Transportation Systems, 2023

> ⚠️ The dataset requires a license agreement. Request access at [cardd-ustc.github.io](https://cardd-ustc.github.io)

**Dataset Split:**

| Split | Images | Instances |
|---|---|---|
| Train | 2,816 | 6,211 |
| Val | 810 | 1,744 |
| Test | 374 | 785 |

---

## 📁 Project Structure

```
car-damage-detection/
├── model/                          # Training & evaluation pipeline
│   ├── scripts/
│   │   ├── convert_annotations.py  # COCO → YOLO format conversion
│   │   ├── check_dataset.py        # Dataset sanity check
│   │   ├── train.py                # Model fine-tuning
│   │   ├── evaluate.py             # Test set evaluation
│   │   ├── visualize.py            # Prediction visualization
│   │   └── show_results.py         # Training curves & charts
│   ├── cardd.yaml                  # Dataset configuration
│   ├── cardd_seg.yaml              # Segmentation dataset config
│   ├── requirements.txt            # Python dependencies
│   └── .env.example                # Environment variables template
├── backend/                        # FastAPI backend
│   ├── app/
│   │   ├── routes/                 # API endpoints
│   │   ├── services/               # YOLO inference & severity scoring
│   │   ├── models/                 # Database models
│   │   └── schemas/                # Pydantic schemas
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                       # React frontend
│   ├── src/
│   │   ├── components/             # UI components
│   │   └── pages/                  # App pages
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml              # Full stack deployment
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- Conda or virtualenv
- Docker & Docker Compose (for full stack)

### 1. Clone the Repository

```bash
git clone https://github.com/abdullahghanem-a11y/car-damage-detection.git
cd car-damage-detection
```

### 2. Create Conda Environment

```bash
conda create -n cardd python=3.10 -y
conda activate cardd
```

### 3. Install Dependencies

```bash
cd model
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and update the paths to match your local setup:

```dotenv
PROJECT_ROOT=C:\path\to\your\cardd_project
DATA_DIR=${PROJECT_ROOT}\data
...
```

### 5. Prepare the Dataset

After downloading CarDD and placing it in `data/`:

```bash
python scripts/convert_annotations.py
python scripts/check_dataset.py
```

---

## 💻 Usage

### Train the Model

```bash
conda activate cardd
cd model
python scripts/train.py
```

### Evaluate on Test Set

```bash
python scripts/evaluate.py
```

### Visualize Predictions

```bash
python scripts/visualize.py
```

### Download Pre-trained Weights

The fine-tuned model is available on HuggingFace:

```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id  = "abdullahg7/cardd-yolov8s",
    filename = "v1.0/best.pt"
)
```

---

## 🐳 Docker Deployment

Run the full stack (frontend + backend + database) with a single command:

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend (React) | http://localhost:80 |
| Backend (FastAPI) | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Database (PostgreSQL) | localhost:5432 |

---

## 📚 References

- Wang, X., Li, W., & Wu, Z. (2023). CarDD: A New Dataset for Vision-Based Car Damage Detection. *IEEE Transactions on Intelligent Transportation Systems*, 24(7), 7202–7214.
- Ultralytics. (2023). YOLOv8. https://github.com/ultralytics/ultralytics

---

## 📄 License

This project is licensed under the **GNU AGPL-3.0 License** in compliance with the YOLOv8 framework by Ultralytics.

> ⚠️ **Dataset Notice:** The CarDD dataset images are subject to Flickr and Shutterstock licensing terms. Proper licensing must be obtained before commercial deployment.