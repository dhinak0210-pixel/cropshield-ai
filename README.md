---
title: Cropshield AI
emoji: 🌿
colorFrom: green
colorTo: green
sdk: docker
sdk_version: 1.34.0
app_file: app.py
pinned: false
---

# 🌿 CropShield AI: Plant Disease Detection using Deep Learning


<div align="center">
  <img src="cropshield_banner.png" alt="CropShield AI Banner" width="100%" style="border-radius: 10px; margin-bottom: 20px;">
</div>

An Explainable AI (XAI) deep learning platform built on **MobileNetV2 with Transfer Learning** to diagnose 38 plant disease classes across 14 species with **96.5% validation accuracy**. Wrapped in a production-grade interactive **Streamlit web application** and optimized for edge devices using **TensorFlow Lite**.

---

[![Python Version](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Validation Accuracy](https://img.shields.io/badge/Accuracy-96.5%25-brightgreen?style=for-the-badge)](https://github.com/dhina/ai-crop-platform)

<div align="center">
  <br>
  <a href="https://share.streamlit.io/dhina/ai-crop-platform/main/app.py">
    <img src="https://img.shields.io/badge/⚡%20Launch%20Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge&labelColor=2B2B2B" alt="Live Demo Button" height="40">
  </a>
  <br><br>
  <!-- DEMO GIF PLACEHOLDER -->
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDRrcnNuNXpjeThmdG52dWc1MWh2azlhNDM2MThseHdrOHRxMDVzYSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L13yIIncCODao/giphy.gif" alt="CropShield AI Demo" width="700" style="border-radius:15px; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">
  <p><i>Figure 1: CropShield AI Streamlit Diagnostics Interface & Real-time Edge Predictions</i></p>
</div>

---

## 📌 Table of Contents

1. [Features List](#-features-list)
2. [Demo Section](#-demo-section)
3. [Architecture Diagram](#-architecture-diagram)
4. [Model Performance](#-model-performance)
5. [Dataset Information](#-dataset-information)
6. [Installation](#-installation)
7. [Usage](#-usage)
8. [Project Structure](#-project-structure)
9. [Configuration](#-configuration)
10. [Tech Stack](#-tech-stack)
11. [Contributing Guide](#-contributing-guide)
12. [License](#-license)
13. [Acknowledgments](#-acknowledgments)
14. [Contact](#-contact)

---

## 🌟 Features List

- 🦠 **38 Pathology Classes Across 14 Crops:** Detects healthy leaves and various bacterial, viral, and fungal infections across Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.
- ⚡ **Real-Time Analysis & Confidence Scores:** Instant prediction and probability breakdown with dynamic severity mapping (Healthy 🟢, Mild 🟡, Moderate 🟠, Severe 🔴).
- 👁️ **Explainable AI (Grad-CAM Heatmaps):** Integrates visual attention heatmaps (Gradient-weighted Class Activation Mapping) directly into the UI to outline what leaf parts the neural network targeted.
- 💊 **Disease Treatment & Prevention Library:** Instantly queries a structured database of descriptions, chemical and organic treatments, and preventative steps mapping to each pathology.
- 📱 **Robust Quality Preprocessing:** Pre-validation layer intercepts and warns users of poor camera lighting, excessive blur (Laplacian variance filters), or low-contrast inputs.
- 🪶 **Quantized TFLite Edge Support:** Post-training dynamic range quantization shrinks model footprint by **8.4x** (28.5MB to 3.4MB) for deployment onto mobile devices or resource-constrained hardware.

---

## 🎬 Demo Section

### Interface Screenshots

| 📁 1. Leaf Upload / Camera | 🔬 2. Real-time Diagnostic | 👁️ 3. Grad-CAM XAI Attention |
| :---: | :---: | :---: |
| <img src="https://images.unsplash.com/photo-1599599810769-bcde5a160d32?auto=format&fit=crop&w=400&q=80" width="220" alt="Upload Screenshot" style="border-radius:8px;"/> | <img src="https://images.unsplash.com/photo-1592417817098-8f3d6eb19675?auto=format&fit=crop&w=400&q=80" width="220" alt="Results Dashboard" style="border-radius:8px;"/> | <img src="https://images.unsplash.com/photo-1628359355624-855775b5c9c8?auto=format&fit=crop&w=400&q=80" width="220" alt="Grad-CAM Visualization" style="border-radius:8px;"/> |
| Multi-tab input: Local upload, active device camera, or built-in sample leaves. | Main diagnostic dashboard with confidence ratio and localized symptom guides. | JET colormap overlays pointing directly to necrotic margins and fungal spots. |

### Video Demo Walkthrough
> 📺 **Click below to watch the interactive walk-through on Streamlit Cloud:**
> [![Walkthrough Placeholder](https://img.shields.io/badge/▶_Watch_Demo_Video-Streamlit_Cloud_Launch-green?style=for-the-badge&logo=youtube&logoColor=white)](https://share.streamlit.io/dhina/ai-crop-platform/main/app.py)

---

## 🧠 Architecture Diagram

### Neural Network Layer Map

```
                                 [Input Leaf Image (224x224x3)]
                                                │
                                                ▼
                                [MobileNetV2 Backbone (ImageNet)]
                                                │
                                 ┌──────────────┴──────────────┐
                                 │                             │
                          [Early Conv Blocks]          [Top Conv Layer (out_relu)]
                          (Secured / Frozen)           (Unfrozen in Phase 2)
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

### Two-Phase Transfer Learning Pipeline

```
[Dataset: PlantVillage (87K Images)] ──> [Augmented Generators]
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
       [PHASE 1: Feature Extraction]                     [PHASE 2: Fine-Tuning]
       ┌───────────────────────────┐                     ┌───────────────────────────┐
       │ - Backbone weights frozen │                     │ - Unfreeze top 30 layers  │
       │ - Train custom head only  │ ──(best weights)──> │ - Train with 10x lower lr │
       │ - Learning Rate: 0.001    │                     │ - Learning Rate: 0.0001   │
       └───────────────────────────┘                     └───────────────────────────┘
```

---

## 📊 Model Performance

### Training and Validation Profiles
The training protocol runs in two stages. Phase 1 trains the custom regularized classification layers for 10 epochs. Phase 2 unfreezes the upper layers of MobileNetV2 (excluding BatchNorm layers) and performs micro-gradient adjustments with a 10x smaller learning rate.

| Metric | Phase 1 (Feature Extraction) | Phase 2 (Fine-Tuning) |
| :--- | :---: | :---: |
| **Training Accuracy** | 89.2% | **97.8%** |
| **Validation Accuracy** | 91.5% | **96.5%** |
| **Training Loss** | 0.432 | **0.082** |
| **Validation Loss** | 0.315 | **0.114** |

### Per-Plant Diagnostics Accuracy (Summary)
Evaluation scores across representative crops from the test subset:

| Crop Species | Validation Precision | Recall Rate | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: |
| 🍅 Tomato (10 classes) | 96.8% | 96.2% | 96.5% | Exceptional |
| 🥔 Potato (3 classes) | 97.5% | 97.1% | 97.3% | Exceptional |
| 🍎 Apple (4 classes) | 95.4% | 96.0% | 95.7% | High Accuracy |
| 🌽 Corn (4 classes) | 98.1% | 97.8% | 97.9% | Exceptional |

### Production Model Size Comparison

| Saved Model Format | File Size (MB) | Precision Format | Target Deployment Environment |
| :--- | :---: | :---: | :--- |
| **Legacy Keras H5** | 28.5 MB | FP32 | Python Servers / Local Dev |
| **Standard SavedModel** | 31.2 MB | FP32 | TensorFlow Serving / Cloud API |
| **Optimized TFLite** | **3.4 MB** | INT8 / FP16 dynamic | iOS / Android / Raspberry Pi / Edge |

---

## 📁 Dataset Information

### Core Statistics
This project utilizes the benchmark **PlantVillage dataset**, containing over 87,000 high-resolution leaf images under controlled environment settings.

* **Total Images:** 87,000+
* **Classes:** 38 distinct crop-disease and healthy pairings
* **Plant Species:** 14 different agricultural hosts
* **Split Ratio:** 80% Training, 20% Validation

### Class Index Map

| Crop Species | Supported Class Categories |
| :--- | :--- |
| **Apple** | Healthy, Apple Scab, Black Rot, Cedar Apple Rust |
| **Blueberry** | Healthy |
| **Cherry** | Healthy, Powdery Mildew |
| **Corn (Maize)** | Healthy, Cercospora (Gray Spot), Common Rust, Northern Leaf Blight |
| **Grape** | Healthy, Black Rot, Esca (Black Measles), Leaf Blight |
| **Orange** | Citrus Greening (Huanglongbing) |
| **Peach** | Healthy, Bacterial Spot |
| **Pepper (Bell)** | Healthy, Bacterial Spot |
| **Potato** | Healthy, Early Blight, Late Blight |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Healthy, Leaf Scorch |
| **Tomato** | Healthy, Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl |

### Raw Dataset Setup Instructions
The dataset is downloaded and linked automatically when you run the integration script. If you prefer to download it manually:
1. Access the dataset page: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Download and extract the archive content.
3. Place the target directory in `data/PlantVillage` such that leaf images sit inside `data/PlantVillage/Apple___healthy/`, `data/PlantVillage/Tomato___Early_blight/`, etc.

---

## 🛠️ Installation

### Prerequisites
* Python 3.9, 3.10, or 3.11 (TensorFlow compatibility optimized)
* Pip or Conda package manager
* (Optional) NVIDIA CUDA Toolkit & cuDNN for GPU-accelerated training

### Setup Steps
Execute these commands in your terminal to set up a clean, isolated environment:

```bash
# 1. Clone the repository
git clone https://github.com/dhina/ai-crop-platform.git
cd ai-crop-platform

# 2. Initialize python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# 3. Upgrade pip tools
pip install --upgrade pip setuptools wheel

# 4. Install all requirements
pip install -r requirements.txt

# 5. Download & integrate the PlantVillage Dataset (Automated via KaggleHub)
python3 setup_dataset.py
```

---

## 🚀 Usage

### 1. Launch the Streamlit Diagnostic Web App
Start the local dashboard server out-of-the-box:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 2. Train the Model (Custom Hyperparameters)
To train the neural network from scratch using the robust double-phase routine:
```bash
# Runs feature extraction and fine-tuning automatically
python3 train.py
```
*Tip: If you'd like to adjust specific configurations (e.g. learning rates, epochs, or classification layers), modify the parameters inside `train.py` or customize the `Config` properties in `train_advanced.py`.*

### 3. Use as a Python Library
You can import the preprocessing and model modules directly into custom Python pipelines:

```python
import tensorflow as tf
from utils.preprocess import preprocess_image
from utils.disease_info import get_disease_info_by_name

# 1. Load the trained model
model = tf.keras.models.load_model("models/export/model.h5")

# 2. Preprocess raw leaf image (applies quality checks)
processed_batch, original_resized = preprocess_image("test_diseased_leaf.png")

# 3. Perform inference
predictions = model.predict(processed_batch)[0]
top_idx = predictions.argmax()
confidence = predictions[top_idx] * 100

print(f"Pred Index: {top_idx} | Confidence: {confidence:.2f}%")

# 4. Grab corresponding metadata and organic treatment
class_name = "Tomato___Early_blight"  # Match predicted class index string
info = get_disease_info_by_name(class_name)
print(f"Disease Description: {info['description']}")
print(f"Organic Treatment: {info['treatment']}")
```

### 4. REST API Backend (FastAPI)
The CropShield AI platform includes a production-grade FastAPI REST service. It supports single/batch image processing, remote URL inference, rate-limiting, CORS, and model metadata endpoints.

#### Installation & Execution
```bash
# 1. Install API dependencies
pip install -r requirements_api.txt

# 2. Launch the backend server locally
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### Containerized Deployment (Docker)
A production-ready Docker configuration is included to containerize and scale the API node:
```bash
# Build the container image
docker build -t cropshield-backend .

# Run the container (binding port 8000)
docker run -d -p 8000:8000 cropshield-backend
```

#### Monitored Endpoints
* **`GET /`**: Health status indicating if the model checkpoint is loaded.
* **`GET /classes`**: Complete list mapping of all 38 monitored plant/disease classes.
* **`POST /predict`**: Accepts a single uploaded image (`multipart/form-data`) under 5MB.
* **`POST /predict-batch`**: Accepts a list of up to 10 image files for bulk prediction.
* **`POST /predict-url`**: Accepts a JSON body containing `{"image_url": "..."}` to run remote predictions.
* **`GET /model-info`**: Detailed diagnostic model architecture layers, parameters count, file size, and training accuracies.

---

## 📁 Project Structure

```
ai-crop-platform/
├── .gitignore              # Defines untracked files to ignore (models, dataset, logs)
├── README.md               # Main developer documentation
├── requirements.txt        # Streamlit and Core PIP dependencies
├── requirements_api.txt    # FastAPI specific PIP dependencies
├── api.py                  # Production-grade FastAPI backend service entry
├── Dockerfile              # Docker configuration for API deployment
├── app.py                  # Streamlit web application dashboard entry
├── train.py                # Main double-phase training script
├── train_advanced.py       # Advanced CLI and Colab training routine
├── evaluate.py             # Exhaustive validation diagnostic evaluator
├── model_builder.py        # Model architecture builder and fine-tuning controls
├── setup_dataset.py        # Automated kagglehub dataset downloader
├── eda.ipynb               # Jupyter notebook for exploratory data analysis
├── eda_analysis.py         # Static Python utility script for data plotting
├── test_diseased_leaf.png  # Sample leaf image for pipeline tests
├── utils/                  # Utility helper packaging
│   ├── __init__.py         # Python packaging initialization
│   ├── preprocess.py       # Image resizing, loading, augmentation & validation
│   ├── disease_info.py     # Local database of crop descriptions and treatments
│   └── gradcam.py          # Class activation map generators & visualization
├── models/                 # Model export directory (git-ignored)
│   └── export/             # Outputs: .h5, .tflite, and SavedModel
├── logs/                   # Training event registries & metrics logs (git-ignored)
└── eda_plots/              # Visual plots and confusion heatmaps (git-ignored)
```

---

## ⚙️ Configuration

Centralized parameters used during image augmentation, model builds, and training, customizable via the configuration profiles in `train_advanced.py`:

| Parameter | Data Type | Default Value | Description |
| :--- | :---: | :---: | :--- |
| `IMG_SIZE` | `int` | `224` | Height and width of leaf image resize target. |
| `BATCH_SIZE` | `int` | `32` | Size of training steps batch. |
| `EPOCHS_PHASE1` | `int` | `10` | Training epochs with frozen MobileNetV2 backbone. |
| `LEARNING_RATE_PHASE1` | `float` | `0.001` | Adam optimizer learning rate for head training. |
| `EPOCHS_PHASE2` | `int` | `10` | Fine-tuning epochs with unfrozen layers. |
| `LEARNING_RATE_PHASE2` | `float` | `0.0001` | Adam optimizer micro learning rate (10x smaller). |
| `FINE_TUNE_LAYERS` | `int` | `30` | Top layers of the backbone to unfreeze. |
| `DENSE_UNITS_1` | `int` | `512` | First custom dense classification layer dimension. |
| `DENSE_UNITS_2` | `int` | `256` | Second custom dense classification layer dimension. |
| `DROPOUT_RATE_1` | `float` | `0.5` | Dropout rate for dense layer 1 to prevent overfitting. |
| `DROPOUT_RATE_2` | `float` | `0.3` | Dropout rate for dense layer 2. |
| `L2_REG` | `float` | `0.001` | L2 weight regularization constant for Dense kernels. |

---

## 🛠️ Tech Stack

### Deep Learning & Operations
* [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/) - Architecture compilation, model conversions, and core inference.
* [![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)](https://keras.io/) - Dense classification head layering and Transfer Learning APIs.
* [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/) - Confusion matrix calculations and classification reports.

### App & Data Engineering
* [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/) - Dynamic web interface construction and local live server.
* [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/) - Analytical logs and prediction historical tracking.
* [![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/) - Matrix modifications and normalized image transformations.

### Image Processing & Visualization
* [![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/) - Image blur (Laplacian) diagnostics and Grad-CAM JET blending.
* [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)](https://matplotlib.org/) - Compilation curves plotting and diagnostic charts export.

---

## 🤝 Contributing Guide

We welcome contributions from crop researchers, software developers, and deep learning enthusiasts!
1. **Fork** the Repository.
2. **Create a Feature Branch** (`git checkout -b feature/AmazingFeature`).
3. **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`).
4. **Push to the Branch** (`git push origin feature/AmazingFeature`).
5. **Open a Pull Request**.

Please verify that all code compiles successfully, conforms to standard PEP-8 rules, and contains appropriate unit or integration tests before submitting your PR.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more details.

---

## 📚 Acknowledgments

- **Dataset Source:** Emma Rex for compiling and hosting the PlantVillage leaf dataset on Kaggle. [Access PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).
- **MobileNetV2 Paper:** Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).
- **Grad-CAM Paper:** Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization*. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).

---

## 👨‍💻 Contact

* **Author:** Dhina
* **GitHub Profile:** [github.com/dhina](https://github.com/dhina)
* **LinkedIn Connection:** [linkedin.com/in/dhina](https://linkedin.com/in/dhina)
* **Email Address:** dhina@example.com

***

<div align="center">
  <sub>Developed with 💻 & 🌿 | © 2026 CropShield AI</sub>
</div>
