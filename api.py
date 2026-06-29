"""
CropShield AI — Production-Grade Plant Disease Detection FastAPI Backend
========================================================================

Usage Examples:
---------------
1. Health Check (GET):
   curl http://localhost:8000/

2. Get Disease Classes (GET):
   curl http://localhost:8000/classes

3. Predict Single Image (POST multipart/form-data):
   curl -X POST -F "file=@test_diseased_leaf.png" http://localhost:8000/predict

4. Predict Batch of Images (POST multipart/form-data):
   curl -X POST -F "files=@test_diseased_leaf.png" -F "files=@test_diseased_leaf.png" http://localhost:8000/predict-batch

5. Predict via Image URL (POST application/json):
   curl -X POST -H "Content-Type: application/json" -d '{"image_url": "https://raw.githubusercontent.com/dhinagaran-dev/ai-crop-platform/main/test_diseased_leaf.png"}' http://localhost:8000/predict-url

6. Get Model Information (GET):
   curl http://localhost:8000/model-info
"""

import os
# Keras 3 compatibility patch for legacy Keras 2 parameters (renorm, quantization_config, etc.)
try:
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization, Dense, Conv2D, DepthwiseConv2D

    def patch_layer_init(layer_class, keys_to_remove):
        original_init = layer_class.__init__
        def patched_init(self, *args, **kwargs):
            for key in keys_to_remove:
                kwargs.pop(key, None)
            original_init(self, *args, **kwargs)
        layer_class.__init__ = patched_init

    patch_layer_init(BatchNormalization, ['renorm', 'renorm_clipping', 'renorm_momentum', 'quantization_config'])
    patch_layer_init(Dense, ['quantization_config'])
    patch_layer_init(Conv2D, ['quantization_config'])
    patch_layer_init(DepthwiseConv2D, ['quantization_config'])
except Exception:
    pass

import io
import time
import re
import json
import logging
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import httpx
from contextlib import asynccontextmanager

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from utils.preprocess import preprocess_image
from utils.disease_info import get_disease_info, get_all_plants

def get_model_target_size(model, default=(224, 224)):
    try:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) == 4:
            return (shape[1], shape[2])
    except Exception:
        pass
    return default

# --- 1. LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("cropshield_api")

# --- 2. CONFIGURATION / SETTINGS ---
class Settings:
    # Model path check order: ENV -> best model -> phase 1 model
    MODEL_PATH: str = os.getenv("MODEL_PATH", "")
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    MAX_BATCH_SIZE: int = int(os.getenv("MAX_BATCH_SIZE", "10"))
    RATE_LIMIT: str = os.getenv("RATE_LIMIT", "10/minute")
    CORS_ORIGINS: List[str] = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",")]

settings = Settings()

# Resolve model path automatically if not explicitly provided
if not settings.MODEL_PATH:
    potential_paths = [
        'models/export/model.h5',
        'models/export/saved_model',
        'models/mobilenetv2_final_best.h5',
        'models/mobilenetv2_phase1_best.h5'
    ]
    for path in potential_paths:
        if os.path.exists(path):
            settings.MODEL_PATH = path
            break

# --- 3. HELPER FUNCTIONS ---
def get_class_names(num_classes: int = 38) -> List[str]:
    """Retrieves class names, falling back to standard PlantVillage classes."""
    # 1. Try loading from model/class_indices.json if the length matches (Primary source of truth)
    try:
        if os.path.exists("model/class_indices.json"):
            with open("model/class_indices.json") as f:
                indices_dict = json.load(f)
                if len(indices_dict) == num_classes:
                    return [indices_dict[str(i)] for i in range(num_classes)]
    except Exception:
        pass

    standard_38_classes = [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", 
        "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot", 
        "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", 
        "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", 
        "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", 
        "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", 
        "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", 
        "Tomato___healthy"
    ]
    
    data_path = 'data/PlantVillage'
    if os.path.exists(data_path):
        dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        if len(dirs) == num_classes:
            return dirs
            
    if num_classes == 38:
        return standard_38_classes
    else:
        if os.path.exists(data_path):
            dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
            if len(dirs) > 0:
                return dirs
        return standard_38_classes[:num_classes]

def get_model_size_mb(path: str) -> float:
    """Returns the size of the model file in Megabytes."""
    try:
        if os.path.exists(path):
            return round(os.path.getsize(path) / (1024 * 1024), 2)
    except Exception as e:
        logger.error(f"Error reading model size: {str(e)}")
    return 0.0

def load_evaluation_metadata() -> Dict[str, Any]:
    """Loads validation metrics and creation date dynamically from log files."""
    metadata = {
        "accuracy": 0.0490,
        "top5_accuracy": 0.1958,
        "evaluation_date": "2026-06-02 19:30:53",
        "dataset": "PlantVillage"
    }
    report_path = "evaluation_results/evaluation_report.txt"
    json_path = "evaluation_results/evaluation_results.json"
    
    if os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                content = f.read()
                date_match = re.search(r"Date Created:\s*(.*)", content)
                if date_match:
                    metadata["evaluation_date"] = date_match.group(1).strip()
        except Exception as e:
            logger.warning(f"Could not parse evaluation date from report: {str(e)}")
            
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                if "global" in data:
                    metadata["accuracy"] = data["global"].get("overall_accuracy", metadata["accuracy"])
                    metadata["top5_accuracy"] = data["global"].get("top5_accuracy", metadata["top5_accuracy"])
        except Exception as e:
            logger.warning(f"Could not parse evaluation metrics from json: {str(e)}")
            
    return metadata

# --- 4. LIFESPAN SYSTEM (MODEL PRELOADING) ---
model_state: Dict[str, Any] = {
    "model": None,
    "classes": []
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup phase
    logger.info("Initializing diagnostic node startup lifecycle...")
    if settings.MODEL_PATH and os.path.exists(settings.MODEL_PATH):
        try:
            logger.info(f"Loading Keras diagnostic checkpoint from: {settings.MODEL_PATH}")
            model_state["model"] = tf.keras.models.load_model(settings.MODEL_PATH)
            num_output_nodes = model_state["model"].output_shape[-1]
            model_state["classes"] = get_class_names(num_output_nodes)
            logger.info("Diagnostic TensorFlow network loaded successfully!")
        except Exception as e:
            logger.error(f"Critical error loading model checkpoint: {str(e)}")
            model_state["classes"] = get_class_names(38)
    else:
        logger.warning(f"No valid model checkpoint file found at '{settings.MODEL_PATH}'. Running in degraded mode.")
        model_state["classes"] = get_class_names(38)
    
    yield
    # Shutdown phase
    logger.info("Shutting down diagnostic node...")
    model_state.clear()

# --- 5. INITIALIZE FASTAPI APP ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="CropShield AI Diagnostic API",
    version="1.0.0",
    description="Production-grade REST API backend serving Deep Learning pathology classifications for crop leaves.",
    lifespan=lifespan
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request Timing & Logging Middleware
@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"IP: {request.client.host if request.client else 'unknown'} | "
        f"Method: {request.method} | "
        f"Path: {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Time: {process_time*1000:.2f}ms"
    )
    response.headers["X-Process-Time-Ms"] = f"{process_time*1000:.2f}"
    return response

# --- 6. PYDANTIC SCHEMAS ---
class PredictURLRequest(BaseModel):
    image_url: HttpUrl

class ClassMapping(BaseModel):
    class_id: int
    class_name: str
    plant: str
    disease: str

class TopPrediction(BaseModel):
    class_name: str
    plant: str
    disease: str
    confidence: float

class PredictionResult(BaseModel):
    success: bool
    predicted_class: Optional[str] = None
    plant: Optional[str] = None
    disease: Optional[str] = None
    confidence: Optional[float] = None
    is_healthy: Optional[bool] = None
    severity: Optional[str] = None
    top_5_predictions: Optional[List[TopPrediction]] = None
    processing_time_ms: int
    error: Optional[str] = None

class BatchSummary(BaseModel):
    total_processed: int
    successful: int
    failed: int
    healthy_count: int
    diseased_count: int
    average_confidence: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    summary: BatchSummary

# --- 7. EXCEPTION HANDLERS ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled system error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )

# --- 8. ENDPOINTS ---

@app.get("/", summary="Health Check")
async def health_check():
    """Health check endpoint to inspect api version and loaded model status."""
    return {
        "api_name": "CropShield AI Diagnostic Node API",
        "version": "1.0.0",
        "status": "healthy",
        "model_loaded": model_state["model"] is not None
    }

@app.get("/classes", response_model=Dict[str, List[ClassMapping]], summary="Get Monitored Disease Classes")
async def get_classes():
    """Returns a full mapping list of all 38 target plant disease classifications."""
    classes_list = model_state["classes"] if model_state["classes"] else get_class_names(38)
    mappings = []
    for idx, c_name in enumerate(classes_list):
        info = get_disease_info(c_name)
        mappings.append(
            ClassMapping(
                class_id=idx,
                class_name=c_name,
                plant=info.get("plant", "Unknown"),
                disease=info.get("disease", "Unknown")
            )
        )
    return {"classes": mappings}

@app.post("/predict", response_model=PredictionResult, summary="Classify Disease in Single Uploaded Leaf")
@limiter.limit(settings.RATE_LIMIT)
async def predict(request: Request, file: UploadFile = File(...)):
    """
    Classify crop disease in a single image.
    Accepts: JPG, JPEG, PNG, or WEBP under 5MB.
    Processes: Resizes to 224x224 and normalizes inputs.
    """
    start_time = time.time()
    
    # 1. Validate extension
    file_ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if file_ext not in ["jpg", "jpeg", "png", "webp"]:
        raise HTTPException(
            status_code=400,
            detail="File extension not supported. Supported extensions: jpg, jpeg, png, webp"
        )
        
    # 2. Validate size (read chunk or read all)
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File size exceeds the maximum limit of 5 Megabytes."
        )
        
    # 3. Predict or mock if model not loaded
    if model_state["model"] is None:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return PredictionResult(
            success=False,
            processing_time_ms=elapsed_ms,
            error="Inference model is currently offline/not loaded."
        )
        
    try:
        image_input = Image.open(io.BytesIO(content))
        model = model_state["model"]
        target_size = get_model_target_size(model)
        processed_img, _ = preprocess_image(image_input, target_size=target_size, validate_leaf=True)
        
        predictions = model_state["model"].predict(processed_img)[0]
        top_indices = np.argsort(predictions)[-5:][::-1]
        
        best_idx = top_indices[0]
        best_conf = float(predictions[best_idx]) * 100
        
        classes_list = model_state["classes"]
        predicted_class = classes_list[best_idx] if best_idx < len(classes_list) else f"Class {best_idx}"
        
        info = get_disease_info(predicted_class)
        plant = info.get("plant", "Unknown")
        disease = info.get("disease", "Unknown")
        is_healthy = (disease.lower() == "healthy" or "healthy" in predicted_class.lower())
        
        # Calculate dynamic severity matching rules from app.py
        severity = "None"
        if not is_healthy:
            if best_conf < 50:
                severity = "Low"
            elif best_conf < 70:
                severity = "Medium"
            elif best_conf < 90:
                severity = "High"
            else:
                severity = "Critical"
                
        # Resolve top 5
        top_5 = []
        for idx in top_indices:
            conf_val = float(predictions[idx]) * 100
            c_name = classes_list[idx] if idx < len(classes_list) else f"Class {idx}"
            info_alt = get_disease_info(c_name)
            top_5.append(
                TopPrediction(
                    class_name=c_name,
                    plant=info_alt.get("plant", "Unknown"),
                    disease=info_alt.get("disease", "Unknown"),
                    confidence=round(conf_val, 2)
                )
            )
            
        elapsed_ms = int((time.time() - start_time) * 1000)
        return PredictionResult(
            success=True,
            predicted_class=predicted_class,
            plant=plant,
            disease=disease,
            confidence=round(best_conf, 2),
            is_healthy=is_healthy,
            severity=severity,
            top_5_predictions=top_5,
            processing_time_ms=elapsed_ms
        )
        
    except ValueError as ve:
        # Preprocessing validation checks failed (blurry, blank, low contrast)
        logger.warning(f"Image validation checks failed: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Prediction layer failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse, summary="Classify Disease in Multiple Uploaded Leaves")
@limiter.limit(settings.RATE_LIMIT)
async def predict_batch(request: Request, files: List[UploadFile] = File(...)):
    """
    Classify crop diseases in a batch of images.
    Accepts: List of up to 10 image files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    if len(files) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum limit of {settings.MAX_BATCH_SIZE} files."
        )
        
    predictions_results = []
    successful = 0
    failed = 0
    healthy_count = 0
    diseased_count = 0
    confidences = []
    
    classes_list = model_state["classes"]
    
    for file in files:
        file_start = time.time()
        try:
            # 1. Validate extension
            file_ext = file.filename.split(".")[-1].lower() if file.filename else ""
            if file_ext not in ["jpg", "jpeg", "png", "webp"]:
                raise ValueError("Unsupported file format. Use: jpg, jpeg, png, webp")
                
            # 2. Validate size
            content = await file.read()
            if len(content) > 5 * 1024 * 1024:
                raise ValueError("File exceeds 5MB size limit.")
                
            if model_state["model"] is None:
                raise ValueError("Inference model is offline.")
                
            image_input = Image.open(io.BytesIO(content))
            model = model_state["model"]
            target_size = get_model_target_size(model)
            processed_img, _ = preprocess_image(image_input, target_size=target_size, validate_leaf=True)
            
            predictions = model_state["model"].predict(processed_img)[0]
            top_indices = np.argsort(predictions)[-5:][::-1]
            
            best_idx = top_indices[0]
            best_conf = float(predictions[best_idx]) * 100
            
            predicted_class = classes_list[best_idx] if best_idx < len(classes_list) else f"Class {best_idx}"
            info = get_disease_info(predicted_class)
            plant = info.get("plant", "Unknown")
            disease = info.get("disease", "Unknown")
            is_healthy = (disease.lower() == "healthy" or "healthy" in predicted_class.lower())
            
            severity = "None"
            if not is_healthy:
                if best_conf < 50:
                    severity = "Low"
                elif best_conf < 70:
                    severity = "Medium"
                elif best_conf < 90:
                    severity = "High"
                else:
                    severity = "Critical"
                    
            top_5 = []
            for idx in top_indices:
                conf_val = float(predictions[idx]) * 100
                c_name = classes_list[idx] if idx < len(classes_list) else f"Class {idx}"
                info_alt = get_disease_info(c_name)
                top_5.append(
                    TopPrediction(
                        class_name=c_name,
                        plant=info_alt.get("plant", "Unknown"),
                        disease=info_alt.get("disease", "Unknown"),
                        confidence=round(conf_val, 2)
                    )
                )
                
            file_elapsed = int((time.time() - file_start) * 1000)
            predictions_results.append(
                PredictionResult(
                    success=True,
                    predicted_class=predicted_class,
                    plant=plant,
                    disease=disease,
                    confidence=round(best_conf, 2),
                    is_healthy=is_healthy,
                    severity=severity,
                    top_5_predictions=top_5,
                    processing_time_ms=file_elapsed
                )
            )
            
            successful += 1
            if is_healthy:
                healthy_count += 1
            else:
                diseased_count += 1
            confidences.append(best_conf)
            
        except Exception as e:
            failed += 1
            file_elapsed = int((time.time() - file_start) * 1000)
            predictions_results.append(
                PredictionResult(
                    success=False,
                    processing_time_ms=file_elapsed,
                    error=str(e)
                )
            )
            
    summary = BatchSummary(
        total_processed=len(files),
        successful=successful,
        failed=failed,
        healthy_count=healthy_count,
        diseased_count=diseased_count,
        average_confidence=round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    )
    
    return BatchPredictionResponse(
        predictions=predictions_results,
        summary=summary
    )

@app.post("/predict-url", response_model=PredictionResult, summary="Classify Disease in Leaf Image downloaded from URL")
@limiter.limit(settings.RATE_LIMIT)
async def predict_url(request: Request, body: PredictURLRequest):
    """Downloads an image from a URL, processes it, and returns the classification."""
    start_time = time.time()
    
    if model_state["model"] is None:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return PredictionResult(
            success=False,
            processing_time_ms=elapsed_ms,
            error="Inference model is currently offline/not loaded."
        )
        
    try:
        url_str = str(body.image_url)
        logger.info(f"Downloading specimen image from URL: {url_str}")
        
        async with httpx.AsyncClient(timeout=12.0) as client:
            try:
                response = await client.get(url_str)
                response.raise_for_status()
            except httpx.HTTPStatusError as hse:
                raise HTTPException(status_code=400, detail=f"URL request failed with status: {hse.response.status_code}")
            except httpx.RequestError as re:
                raise HTTPException(status_code=400, detail=f"Failed to communicate with image URL: {str(re)}")
                
        # Validate downloaded content type
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"URL did not return an image content type (got: {content_type}).")
            
        content = response.content
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Downloaded image exceeds 5MB size limit.")
            
        # Run inference
        image_input = Image.open(io.BytesIO(content))
        model = model_state["model"]
        target_size = get_model_target_size(model)
        processed_img, _ = preprocess_image(image_input, target_size=target_size, validate_leaf=True)
        
        predictions = model_state["model"].predict(processed_img)[0]
        top_indices = np.argsort(predictions)[-5:][::-1]
        
        best_idx = top_indices[0]
        best_conf = float(predictions[best_idx]) * 100
        
        classes_list = model_state["classes"]
        predicted_class = classes_list[best_idx] if best_idx < len(classes_list) else f"Class {best_idx}"
        
        info = get_disease_info(predicted_class)
        plant = info.get("plant", "Unknown")
        disease = info.get("disease", "Unknown")
        is_healthy = (disease.lower() == "healthy" or "healthy" in predicted_class.lower())
        
        # Calculate dynamic severity
        severity = "None"
        if not is_healthy:
            if best_conf < 50:
                severity = "Low"
            elif best_conf < 70:
                severity = "Medium"
            elif best_conf < 90:
                severity = "High"
            else:
                severity = "Critical"
                
        top_5 = []
        for idx in top_indices:
            conf_val = float(predictions[idx]) * 100
            c_name = classes_list[idx] if idx < len(classes_list) else f"Class {idx}"
            info_alt = get_disease_info(c_name)
            top_5.append(
                TopPrediction(
                    class_name=c_name,
                    plant=info_alt.get("plant", "Unknown"),
                    disease=info_alt.get("disease", "Unknown"),
                    confidence=round(conf_val, 2)
                )
            )
            
        elapsed_ms = int((time.time() - start_time) * 1000)
        return PredictionResult(
            success=True,
            predicted_class=predicted_class,
            plant=plant,
            disease=disease,
            confidence=round(best_conf, 2),
            is_healthy=is_healthy,
            severity=severity,
            top_5_predictions=top_5,
            processing_time_ms=elapsed_ms
        )
        
    except ValueError as ve:
        logger.warning(f"Image validation checks failed: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL Prediction layer failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info", summary="Get Diagnostics Network Meta Information")
async def get_model_info():
    """Returns detailed architecture configurations, size, and training evaluation data."""
    metadata = load_evaluation_metadata()
    model_size_mb = get_model_size_mb(settings.MODEL_PATH)
    
    # Custom head architecture description
    custom_head_layers = [
        "GlobalAveragePooling2D",
        "BatchNormalization",
        "Dense(512, activation='relu', kernel_regularizer=L2(0.001))",
        "Dropout(0.5)",
        "BatchNormalization",
        "Dense(256, activation='relu', kernel_regularizer=L2(0.001))",
        "Dropout(0.3)",
        "Dense(38, activation='softmax')"
    ]
    
    num_classes = len(model_state["classes"]) if model_state["classes"] else 38
    
    return {
        "model_file_name": os.path.basename(settings.MODEL_PATH) if settings.MODEL_PATH else "N/A",
        "model_file_size_mb": model_size_mb,
        "model_loaded": model_state["model"] is not None,
        "architecture": {
            "backbone": "MobileNetV2 (pretrained on ImageNet, top 30 layers fine-tuned)",
            "input_shape": [224, 224, 3],
            "custom_head_layers": custom_head_layers,
            "classes_count": num_classes
        },
        "training_metadata": {
            "validation_accuracy": round(metadata["accuracy"] * 100, 2),
            "top_5_validation_accuracy": round(metadata["top5_accuracy"] * 100, 2),
            "training_completion_date": metadata["evaluation_date"],
            "dataset_origin": metadata["dataset"]
        }
    }

# --- 9. LOCAL RUN OVERRIDE ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting uvicorn server on {settings.HOST}:{settings.PORT}")
    uvicorn.run("api:app", host=settings.HOST, port=settings.PORT, reload=True)
