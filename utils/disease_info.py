"""
CropShield AI — Comprehensive Plant Disease Diagnostic Database
==============================================================
Contains comprehensive agricultural science mappings for all 38 classes of the
PlantVillage dataset. Each class contains detailed profiles (symptoms, treatments,
preventions, pathogens, severity profiles, spreading models, and economic impacts).

Includes robust normalization routines and search/filtering utilities for dynamic UIs.
"""

import re
import json
import os
from typing import Dict, Any, List

# ==========================================
# 📊 COMPREHENSIVE 38-CLASS DATABASE
# ==========================================
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_JSON_PATH = os.path.join(os.path.dirname(_CURRENT_DIR), "data", "disease_info.json")

DISEASE_INFO = {}
if os.path.exists(_JSON_PATH):
    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as _f:
            DISEASE_INFO = json.load(_f)
    except Exception as _e:
        pass

# ==========================================
# 🔍 CLASS NAME NORMALIZATION ENGINE
# ==========================================
def normalize_class_name(class_name: str) -> str:
    """
    Cleans and standardizes variations of class names from dataset directories,
    Kaggle models, or raw files, to match the standard 38 keys in the database.
    """
    if not class_name:
        return "Tomato___healthy"
        
    # Extract plant & disease by checking separators or known plant name prefixes
    if "___" in class_name:
        parts = class_name.split("___", 1)
        plant = parts[0]
        disease = parts[1]
    elif "__" in class_name:
        parts = class_name.split("__", 1)
        plant = parts[0]
        disease = parts[1]
    else:
        plant_names = ["Tomato", "Potato", "Pepper", "Apple", "Grape", "Peach", "Strawberry", "Cherry", "Corn"]
        found = False
        for p in plant_names:
            if class_name.lower().startswith(p.lower()):
                plant = p
                disease = class_name[len(p):].lstrip("_")
                found = True
                break
        if not found:
            parts = class_name.split("_", 1)
            plant = parts[0]
            disease = parts[1] if len(parts) > 1 else "healthy"
            
    plant = plant.strip()
    disease = disease.strip()
    
    # Standardize crop name mappings
    plant_lower = plant.lower()
    if "pepper" in plant_lower:
        plant = "Pepper"
    elif "corn" in plant_lower:
        plant = "Corn"
    elif "cherry" in plant_lower:
        plant = "Cherry"
    elif "potato" in plant_lower:
        plant = "Potato"
    elif "tomato" in plant_lower:
        plant = "Tomato"
    elif "apple" in plant_lower:
        plant = "Apple"
    elif "grape" in plant_lower:
        plant = "Grape"
    elif "strawberry" in plant_lower:
        plant = "Strawberry"
    elif "peach" in plant_lower:
        plant = "Peach"
        
    # Standardize disease name mappings
    dis_lower = disease.lower()
    if "healthy" in dis_lower:
        disease = "healthy"
    elif "scab" in dis_lower:
        disease = "Apple_scab" if plant == "Apple" else "scab"
    elif "black rot" in dis_lower:
        disease = "Black_rot"
    elif "cedar apple rust" in dis_lower or "rust" in dis_lower:
        disease = "Cedar_apple_rust" if plant == "Apple" else ("Common_rust" if plant == "Corn" else "rust")
    elif "powdery mildew" in dis_lower:
        disease = "Powdery_mildew"
    elif "cercospora" in dis_lower or "gray leaf spot" in dis_lower:
        disease = "Cercospora_leaf_spot"
    elif "northern leaf blight" in dis_lower or "northern" in dis_lower:
        disease = "Northern_Leaf_Blight"
    elif "esca" in dis_lower:
        disease = "Esca"
    elif "leaf blight" in dis_lower:
        disease = "Leaf_blight"
    elif "haunglongbing" in dis_lower or "greening" in dis_lower:
        disease = "Haunglongbing"
    elif "bacterial spot" in dis_lower:
        disease = "Bacterial_spot"
    elif "early blight" in dis_lower:
        disease = "Early_blight"
    elif "late blight" in dis_lower:
        disease = "Late_blight"
    elif "leaf mold" in dis_lower:
        disease = "Leaf_Mold"
    elif "septoria" in dis_lower:
        disease = "Septoria_leaf_spot"
    elif "spider mites" in dis_lower or "mites" in dis_lower:
        disease = "Spider_mites"
    elif "target spot" in dis_lower:
        disease = "Target_Spot"
    elif "yellow leaf curl" in dis_lower or "yellowleaf" in dis_lower:
        disease = "Tomato_Yellow_Leaf_Curl_Virus"
    elif "mosaic virus" in dis_lower:
        disease = "Tomato_mosaic_virus"
    elif "scorch" in dis_lower:
        disease = "Leaf_scorch"
        
    # Construct standard database key
    db_key = f"{plant}___{disease}"
    
    # Fallback to key exact matching
    if db_key in DISEASE_INFO:
        return db_key
        
    # Check lowercase mapping fallback
    db_key_lower = db_key.lower()
    for actual_key in DISEASE_INFO.keys():
        if actual_key.lower() == db_key_lower:
            return actual_key
            
    # Try fuzzy crop-to-disease mappings
    for actual_key in DISEASE_INFO.keys():
        if plant in actual_key and disease.split("_")[0] in actual_key:
            return actual_key
            
    return "Tomato___healthy"

# ==========================================
# 🛠️ HELPER FUNCTIONS IMPLEMENTATION
# ==========================================
def get_disease_info(class_name: str) -> Dict[str, Any]:
    """
    Retrieves the comprehensive disease information dictionary for a given class name.
    Automatically normalizes variations in formatting.
    """
    normalized_key = normalize_class_name(class_name)
    default_info = {
        "plant": "Crop",
        "disease": "Unknown Pathology",
        "scientific_name": "N/A",
        "cause": "unknown",
        "symptoms": ["Visible signs of crop distress on the foliage canopy."],
        "treatment": ["Please isolate the crop and consult a local agricultural extension office."],
        "prevention": ["Maintain standard weeding, ventilation, and watering schedules."],
        "severity": "None",
        "spreading": "Spreading vectors are documented locally.",
        "favorable_conditions": "Favorable environmental conditions are being mapped.",
        "economic_impact": "Economic loss potential varies across regions."
    }
    return DISEASE_INFO.get(normalized_key, default_info)

# Legacy alias for Streamlit backward compatibility
def get_disease_info_by_name(class_name: str) -> Dict[str, Any]:
    """Dynamically routes name-based queries from UIs."""
    return get_disease_info(class_name)

def get_severity_color(severity: str) -> str:
    """
    Maps a severity tier to a specific hex color code for UI cards and badges.
    """
    severity_lower = str(severity).lower()
    colors = {
        "none": "#2ecc71",     # Vibrant Green (Healthy)
        "low": "#f1c40f",      # Sunflower Yellow
        "medium": "#e67e22",   # Orange
        "high": "#e74c3c",     # Light Red / Tomato
        "critical": "#9b59b6"  # Royal Purple
    }
    return colors.get(severity_lower, "#95a5a6") # Neutral Gray default

def get_severity_emoji(severity: str) -> str:
    """
    Maps a severity level to a descriptive badge emoji.
    """
    severity_lower = str(severity).lower()
    emojis = {
        "none": "🟢 Healthy",
        "low": "🟡 Low / Mild",
        "medium": "🟠 Moderate",
        "high": "🔴 Severe",
        "critical": "🚨 Critical / Outbreak"
    }
    return emojis.get(severity_lower, "⚪ Unknown")

def get_all_plants() -> List[str]:
    """
    Returns a sorted list of all unique plant common names in the database.
    """
    plants = set()
    for info in DISEASE_INFO.values():
        plants.add(info["plant"])
    return sorted(list(plants))

def get_diseases_for_plant(plant_name: str) -> List[Dict[str, Any]]:
    """
    Returns a list of all disease information profiles associated with a specific plant.
    """
    if not plant_name:
        return []
        
    target_plant = plant_name.strip().lower()
    results = []
    
    for key, info in DISEASE_INFO.items():
        if info["plant"].lower() == target_plant:
            # Inject key for easy mapping in loop
            record = info.copy()
            record["class_key"] = key
            results.append(record)
            
    return results

def search_diseases(keyword: str) -> List[Dict[str, Any]]:
    """
    Performs a full-text search across plant names, diseases, pathogens, and symptoms,
    returning matching profiles. Useful for agricultural lookup searches.
    """
    if not keyword:
        return []
        
    query = keyword.strip().lower()
    results = []
    
    for key, info in DISEASE_INFO.items():
        # Search criteria
        searchable_text = f"{info['plant']} {info['disease']} {info['scientific_name']} {info['cause']} " \
                          f"{' '.join(info['symptoms'])} {' '.join(info['treatment'])} {' '.join(info['prevention'])}"
        
        if query in searchable_text.lower():
            record = info.copy()
            record["class_key"] = key
            results.append(record)
            
    return results
