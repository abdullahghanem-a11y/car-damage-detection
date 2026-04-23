# 🚗 Car Damage Detection System

A vision-based car damage detection system built with **YOLOv8** and fine-tuned on the **CarDD dataset**. The system automatically localizes and classifies six types of car damage from images, achieving **71.67% mAP50** on the test set — surpassing 8 out of 10 models from the original benchmark paper.

> 📦 **Model weights hosted on HuggingFace:** [abdullahg7/cardd-yolov8s](https://huggingface.co/abdullahg7/cardd-yolov8s)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Roadmap](#roadmap)

---

## 🔍 Overview

This project fine-tunes a pre-trained **YOLOv8s** model on the CarDD dataset for real-world car damage detection. It is designed as a full software product with a planned backend (FastAPI), database (PostgreSQL), and frontend (React).

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

| Metric | Score |
|---|---|
| mAP@0.5 | **71.67%** |
| mAP@0.5:0.95 | **55.43%** |
| Precision | 75.11% |
| Recall | 67.52% |
| Inference Speed | 7.7ms / image |
| Model Size | 22.5 MB |
| Training Time | ~1 hour (RTX 3050) |

### Per-Class Performance

| Class | mAP50 |
|---|---|
| Glass Shatter | 99.0% |
| Tire Flat | 90.5% |
| Lamp Broken | 85.5% |
| Scratch | 59.6% |
| Dent | 57.8% |
| Crack | 37.7% |

### Benchmark Comparison

| Model | mAP50 | vs Ours |
|---|---|---|
| Mask R-CNN ResNet-50 | 66.3% | ✅ +5.4% |
| Cascade Mask R-CNN ResNet-50 | 64.7% | ✅ +7.0% |
| GCNet ResNet-50 | 66.4% | ✅ +5.3% |
| HTC ResNet-50 | 68.1% | ✅ +3.7% |
| DCN ResNet-50 | 70.9% | ✅ +0.8% |
| Mask R-CNN ResNet-101 | 67.7% | ✅ +4.0% |
| HTC ResNet-101 | 68.4% | ✅ +3.3% |
| DCN ResNet-101 | 69.8% | ✅ +1.9% |
| **YOLOv8s (Ours)** | **71.7%** | 🔥 |
| DCN+ ResNet-50 | 77.4% | ❌ -5.7% |
| DCN+ ResNet-101 | 78.8% | ❌ -7.1% |

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
├── model/                        # Training & evaluation pipeline
│   ├── scripts/
│   │   ├── convert_annotations.py  # COCO → YOLO format conversion
│   │   ├── check_dataset.py        # Dataset sanity check
│   │   ├── train.py                # Model fine-tuning
│   │   ├── evaluate.py             # Test set evaluation
│   │   ├── visualize.py            # Prediction visualization
│   │   └── show_results.py         # Training curves & charts
│   ├── cardd.yaml                  # Dataset configuration
│   ├── requirements.txt            # Python dependencies
│   └── .env.example                # Environment variables template
├── backend/                      # FastAPI backend (coming soon)
├── frontend/                     # React frontend (coming soon)
├── database/                     # PostgreSQL schema (coming soon)
├── docker-compose.yml            # Docker setup (coming soon)
└── .gitignore
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support
- Conda or virtualenv

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
```env
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

### Generate Result Charts
```bash
python scripts/show_results.py
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

## 🗺️ Roadmap

- [x] **Phase 1** — Model training & evaluation
- [ ] **Phase 2** — FastAPI backend with inference API
- [ ] **Phase 3** — PostgreSQL database integration
- [ ] **Phase 4** — React frontend
- [ ] **Phase 5** — Docker containerization

---

## 📚 References

- Wang, X., Li, W., & Wu, Z. (2023). CarDD: A New Dataset for Vision-Based Car Damage Detection. *IEEE Transactions on Intelligent Transportation Systems*, 24(7), 7202–7214.
- Ultralytics. (2023). YOLOv8. https://github.com/ultralytics/ultralytics

---

## 📄 License

This project is licensed under the **GNU AGPL-3.0 License** in compliance 
with the YOLOv8 framework by Ultralytics. This means you are free to use, 
modify, and deploy this software commercially as long as the source code 
remains publicly available.

> ⚠️ **Dataset Notice:** The CarDD dataset images are subject to Flickr 
> and Shutterstock licensing terms. Proper licensing must be obtained 
> before commercial deployment.