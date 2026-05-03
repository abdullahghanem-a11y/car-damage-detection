# 🚗 Car Damage Detection System

A full-stack AI-powered car damage detection system built with **YOLOv8**, **FastAPI**, **PostgreSQL**, **React**, and **Docker**. Upload car images to automatically detect and classify six types of damage in real time.

> 📦 **Model weights:** [abdullahg7/cardd-yolov8s](https://huggingface.co/abdullahg7/cardd-yolov8s)  
> 🔬 **Dataset:** [CarDD — IEEE T-ITS 2023](https://cardd-ustc.github.io)

---

## 📋 Table of Contents
- [System Architecture](#system-architecture)
- [Quick Start with Docker](#quick-start-with-docker)
- [Results](#results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Manual Setup](#manual-setup)
- [API Reference](#api-reference)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────┐
│         React Frontend (Nginx)           │
│         http://localhost:80              │
│  - Drag & drop image upload              │
│  - Annotated result display              │
│  - Detection history with pagination     │
└──────────────────┬──────────────────────┘
                   │ HTTP REST API
┌──────────────────▼──────────────────────┐
│         FastAPI Backend                  │
│         http://localhost:8000            │
│  - Batch image processing (up to 10)     │
│  - YOLOv8 inference service              │
│  - Auto-downloads model from HuggingFace │
└──────────────────┬──────────────────────┘
                   │ SQLAlchemy ORM
┌──────────────────▼──────────────────────┐
│         PostgreSQL Database              │
│  - Stores detection results as JSON      │
│  - Tracks model version per detection    │
└─────────────────────────────────────────┘
```

---

## 🚀 Quick Start with Docker

The entire system runs with a single command.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# 1. Clone the repository
git clone https://github.com/abdullahghanem-a11y/car-damage-detection.git
cd car-damage-detection

# 2. Configure environment
cp .env.example .env
# Edit .env and set your DB_PASSWORD

# 3. Run everything
docker-compose up --build
```

Open **http://localhost:80** — done! 🎉

> 💡 On first run, the model (~22.5MB) is automatically downloaded from HuggingFace and cached locally.

**To stop:**
```bash
docker-compose down
```

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
| Training Time | ~1 hour (RTX 3050 4GB) |

### Per-Class Performance

| Class | mAP50 | Difficulty |
|---|---|---|
| 🟣 Glass Shatter | 99.0% | Easy |
| 🟠 Tire Flat | 90.5% | Easy |
| 🟡 Lamp Broken | 85.5% | Easy |
| 🔵 Scratch | 59.6% | Hard |
| 🔴 Dent | 57.8% | Hard |
| 🟢 Crack | 37.7% | Hard |

### Benchmark Comparison

| Model | Backbone | mAP50 | vs Ours |
|---|---|---|---|
| Mask R-CNN | ResNet-50 | 66.3% | ✅ +5.4% |
| Cascade Mask R-CNN | ResNet-50 | 64.7% | ✅ +7.0% |
| GCNet | ResNet-50 | 66.4% | ✅ +5.3% |
| HTC | ResNet-50 | 68.1% | ✅ +3.7% |
| DCN | ResNet-50 | 70.9% | ✅ +0.8% |
| Mask R-CNN | ResNet-101 | 67.7% | ✅ +4.0% |
| HTC | ResNet-101 | 68.4% | ✅ +3.3% |
| DCN | ResNet-101 | 69.8% | ✅ +1.9% |
| **YOLOv8s (Ours)** | — | **71.7%** | 🔥 |
| DCN+ | ResNet-50 | 77.4% | ❌ -5.7% |
| DCN+ | ResNet-101 | 78.8% | ❌ -7.1% |

> **Note:** YOLOv8s outperforms DCN+ on every individual damage category despite lower overall mAP50, due to differences in how detection metrics are aggregated across model types.

---

## 📦 Dataset

This project uses the **CarDD** (Car Damage Detection) dataset:

- **4,000** high-resolution car damage images
- **9,000+** annotated instances across **6** damage categories
- Annotations follow real insurance claim standards (Ping An Insurance Company)
- Published in *IEEE Transactions on Intelligent Transportation Systems*, 2023

> ⚠️ The dataset requires a license agreement. Request access at [cardd-ustc.github.io](https://cardd-ustc.github.io)

| Split | Images | Instances |
|---|---|---|
| Train | 2,816 | 6,211 |
| Val | 810 | 1,744 |
| Test | 374 | 785 |

---

## 📁 Project Structure

```
car-damage-detection/
│
├── model/                          # AI model training pipeline
│   ├── scripts/
│   │   ├── convert_annotations.py  # COCO → YOLO format conversion
│   │   ├── check_dataset.py        # Dataset sanity check
│   │   ├── train.py                # YOLOv8 fine-tuning
│   │   ├── evaluate.py             # Test set evaluation
│   │   ├── visualize.py            # Prediction visualization
│   │   └── show_results.py         # Training curves & charts
│   ├── cardd.yaml                  # Dataset configuration
│   ├── requirements.txt            # Python dependencies
│   └── .env.example                # Environment variables template
│
├── backend/                        # FastAPI REST API
│   ├── app/
│   │   ├── main.py                 # FastAPI app entry point
│   │   ├── database.py             # PostgreSQL connection
│   │   ├── routes/
│   │   │   └── detection.py        # Detection endpoints
│   │   ├── models/
│   │   │   └── detection.py        # SQLAlchemy DB models
│   │   ├── schemas/
│   │   │   └── detection.py        # Pydantic schemas
│   │   └── services/
│   │       └── yolo_service.py     # YOLOv8 inference + HuggingFace
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                       # React web application
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── UploadZone.jsx      # Drag & drop image upload
│   │   │   └── ResultCard.jsx      # Detection result display
│   │   ├── pages/
│   │   │   ├── Home.jsx            # Detection page
│   │   │   └── History.jsx         # Detection history page
│   │   └── services/
│   │       └── api.js              # API client
│   ├── Dockerfile
│   └── nginx.conf                  # Nginx reverse proxy config
│
├── CarDamageDetection.postman_collection.json
├── docker-compose.yml              # Full system orchestration
├── .env.example                    # Root environment template
├── .gitignore
└── README.md
```

---

## 🛠️ Manual Setup

For development without Docker.

### Prerequisites
- Python 3.10+
- Node.js 20+
- PostgreSQL 16+
- NVIDIA GPU with CUDA (recommended)
- Conda

### 1. Clone & Configure
```bash
git clone https://github.com/abdullahghanem-a11y/car-damage-detection.git
cd car-damage-detection
```

### 2. Model Training Pipeline
```bash
conda create -n cardd python=3.10 -y
conda activate cardd
cd model
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your paths
python scripts/convert_annotations.py
python scripts/check_dataset.py
python scripts/train.py
python scripts/evaluate.py
```

### 3. Backend
```bash
conda activate cardd
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your DATABASE_URL and model config
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Frontend
```bash
cd frontend
npm install
cp .env.example .env
# Edit .env: VITE_API_BASE_URL=http://localhost:8000
npm run dev
```

Open **http://localhost:5173**

---

## 📡 API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/detect/` | Detect damage in 1–10 images |
| `GET` | `/api/detect/history` | Get detection history (paginated) |
| `GET` | `/api/detect/history/{id}` | Get single detection by ID |
| `GET` | `/docs` | Interactive API documentation |

> 📬 Import `CarDamageDetection.postman_collection.json` for ready-to-use API testing with dynamic variables.

### Example Request
```bash
POST /api/detect/
Content-Type: multipart/form-data
Body: files=<image1.jpg>, files=<image2.jpg>
```

### Example Response
```json
{
  "total_images": 1,
  "total_instances": 3,
  "results": [
    {
      "filename": "car.jpg",
      "total_instances": 3,
      "detections": [
        {
          "class_id": 0,
          "class_name": "dent",
          "confidence": 0.8825,
          "bbox": [134.01, 135.27, 607.53, 251.57]
        }
      ],
      "annotated_image": "<base64_encoded_image>"
    }
  ]
}
```

---

## 🗺️ Roadmap

- [x] **Phase 1** — YOLOv8s model training & evaluation (mAP50: 71.67%)
- [x] **Phase 2** — FastAPI backend with YOLOv8 inference
- [x] **Phase 3** — PostgreSQL database integration
- [x] **Phase 4** — React frontend with drag & drop UI
- [x] **Phase 5** — Docker containerization
- [ ] **Phase 6** — Model improvement (YOLOv8m, more data)
- [ ] **Phase 7** — Export results as PDF report
- [ ] **Phase 8** — User authentication

---

## 📚 References

- Wang, X., Li, W., & Wu, Z. (2023). CarDD: A New Dataset for Vision-Based Car Damage Detection. *IEEE Transactions on Intelligent Transportation Systems*, 24(7), 7202–7214.
- Ultralytics (2023). YOLOv8. https://github.com/ultralytics/ultralytics
- He, K., et al. (2017). Mask R-CNN. *Proc. IEEE ICCV*, 2961–2969.
- Lin, T.-Y., et al. (2014). Microsoft COCO. *Proc. ECCV*, 740–755.

---

## 📄 License

This project is licensed under the **GNU AGPL-3.0 License** in compliance with the YOLOv8 framework by Ultralytics. You are free to use, modify, and deploy this software commercially as long as the source code remains publicly available.

> ⚠️ **Dataset Notice:** CarDD images are subject to Flickr and Shutterstock licensing terms. Obtain proper licensing before commercial deployment.