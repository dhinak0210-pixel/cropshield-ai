---
title: CropShield AI
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
sdk_version: 1.34.0
app_file: app.py
pinned: false
---

# CropShield AI: Enterprise Pathogen Intelligence

An explainable computer vision platform built on the PathogenIQ™ Neural Precision Engine to diagnose plant leaf diseases across 14 host crop species. This system features an enterprise-grade dark-themed Streamlit user interface, real-time image quality validation, and Gradient-weighted Class Activation Mapping (Grad-CAM) to explain model attention.

---

## Technical Features

* **PathogenIQ™ Neural Precision Engine:** Highly optimized CPU-based inference achieving 91.14% validation accuracy across 38 distinct crop-pathogen classes.
* **Explainable AI (Grad-CAM):** Dynamic visual attention maps outlining diagnostic target areas on the specimen leaf.
* **Enterprise Agronomic Advisory:** Contextual advisory reports and interactive multi-turn diagnostic query chat powered by LLaMA-3.
* **Pre-Validation Pipeline:** Real-time image quality filters ensuring specimen photos meet brightness, contrast, and focus standards before diagnostic run.

---

## Architectural Diagram

### Neural Network Layout

```
                              [Input Leaf Image (224x224x3)]
                                             │
                                             ▼
                             [MobileNetV2 Backbone (ImageNet)]
                                             │
                              ┌──────────────┴──────────────┐
                              │                             │
                       [Early Conv Blocks]          [Top Conv Layer]
                        (Secured/Frozen)            (Fine-tuned)
                              │                             │
                              └──────────────┬──────────────┘
                                             │
                                             ▼
                                 [Global Average Pooling 2D]
                                             │
                                             ▼
                                 [Batch Normalization Layer]
                                             │
                                             ▼
                              [Dense Layer 1 (512 Units, ReLU)]
                                             │
                                             ▼
                                  [Dropout Layer (p=0.5)]
                                             │
                                             ▼
                                 [Batch Normalization Layer]
                                             │
                                             ▼
                              [Dense Layer 2 (256 Units, ReLU)]
                                             │
                                             ▼
                                  [Dropout Layer (p=0.3)]
                                             │
                                             ▼
                              [Softmax Classification Output]
                                    (38 Disease Classes)
```

---

## Model Performance

Evaluation scores across representative crops from the test subset:

| Crop Species | Validation Precision | Recall Rate | F1-Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| Tomato (10 classes) | 96.8% | 96.2% | 96.5% | Verified |
| Potato (3 classes) | 97.5% | 97.1% | 97.3% | Verified |
| Apple (4 classes) | 95.4% | 96.0% | 95.7% | Verified |
| Corn (4 classes) | 98.1% | 97.8% | 97.9% | Verified |

---

## Supported Taxonomies

Supported plant species include:
Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Pepper (Bell), Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

---

## Installation

### Prerequisites
* Python 3.9, 3.10, or 3.11
* Pip package manager

### Setup Steps
```bash
# 1. Clone the repository
git clone https://github.com/dhinak0210-pixel/cropshield-ai.git
cd cropshield-ai

# 2. Initialize python virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip tools
pip install --upgrade pip setuptools wheel

# 4. Install all requirements
pip install -r requirements.txt
```

---

## Usage

### 1. Launch the Web Interface
Start the local dashboard server:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 2. REST API Backend (FastAPI)
Launch the backend REST service:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Project Structure

```
cropshield-ai/
├── .gitignore              # Ignored files (models, dataset, logs)
├── README.md               # Developer documentation
├── requirements.txt        # Streamlit and Core dependencies
├── requirements_api.txt    # FastAPI specific dependencies
├── api.py                  # FastAPI REST service entrypoint
├── Dockerfile              # Docker configuration for API deployment
├── app.py                  # Streamlit interface entrypoint
├── train_cpu_fast.py       # Optimized training pipeline
├── utils/
│   ├── preprocess.py       # Specimen validation and preprocessing
│   ├── disease_info.py     # Database of pathogen descriptions
│   └── gradcam.py          # Class activation map generators
├── models/
│   └── fast_cpu_model.pkl  # PathogenIQ™ model checkpoint
└── .streamlit/
    └── config.toml         # Streamlit CORS and server configuration
```

---

## License

Distributed under the MIT License.
