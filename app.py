# ============================================
# app.py - COMPLETE APP WITH LLM INTEGRATION
# ============================================

import os
# Keras 3 compatibility patch for legacy Keras 2 BatchNormalization parameters
try:
    import tensorflow as tf
    from tensorflow.keras.layers import BatchNormalization
    original_bn_init = BatchNormalization.__init__
    def patched_bn_init(self, *args, **kwargs):
        kwargs.pop('renorm', None)
        kwargs.pop('renorm_clipping', None)
        kwargs.pop('renorm_momentum', None)
        original_bn_init(self, *args, **kwargs)
    BatchNormalization.__init__ = patched_bn_init
except Exception:
    pass

import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image
from dotenv import load_dotenv

# Local imports
from llm.llm_client import get_llm_client
from llm.disease_advisor import PlantDiseaseAdvisor
from llm.chat_handler import ChatHandler

load_dotenv()

# ─── PAGE CONFIG ─────────────────────────────
st.set_page_config(
    page_title="🌿 PlantDoc AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(
            135deg, #1a5276, #27ae60
        );
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
    }

    /* Prediction card */
    .pred-card {
        background: linear-gradient(
            135deg, #2c3e50, #3498db
        );
        border-radius: 15px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }

    /* Disease info card */
    .info-card {
        background: #f8f9fa;
        border-left: 5px solid #27ae60;
        border-radius: 8px;
        padding: 15px;
        margin: 8px 0;
        color: #1c2833 !important;
    }
    .info-card h1, .info-card h2, .info-card h3, .info-card p, .info-card li {
        color: #1c2833 !important;
    }

    /* Severity badges */
    .badge-none     {background:#2ecc71; color:white;
                     padding:5px 12px; border-radius:15px;}
    .badge-low      {background:#f39c12; color:white;
                     padding:5px 12px; border-radius:15px;}
    .badge-medium   {background:#e67e22; color:white;
                     padding:5px 12px; border-radius:15px;}
    .badge-high     {background:#e74c3c; color:white;
                     padding:5px 12px; border-radius:15px;}
    .badge-critical {background:#7b241c; color:white;
                     padding:5px 12px; border-radius:15px;}

    /* Chat messages */
    .chat-user {
        background: #d5e8d4;
        border-radius: 15px 15px 5px 15px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-left: 20%;
        color: #1c2833 !important;
    }
    .chat-user p, .chat-user span, .chat-user h1, .chat-user h2, .chat-user h3 {
        color: #1c2833 !important;
    }
    .chat-ai {
        background: #dae8fc;
        border-radius: 15px 15px 15px 5px;
        padding: 12px 16px;
        margin: 8px 0;
        margin-right: 20%;
        color: #1c2833 !important;
    }
    .chat-ai p, .chat-ai span, .chat-ai h1, .chat-ai h2, .chat-ai h3 {
        color: #1c2833 !important;
    }

    /* AI response box */
    .ai-response {
        background: #eafaf1;
        border: 1px solid #27ae60;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: #1c2833 !important;
    }
    .ai-response h1, .ai-response h2, .ai-response h3, .ai-response p, .ai-response li, .ai-response strong {
        color: #1c2833 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOAD RESOURCES ──────────────────────────
@st.cache_resource
def load_model():
    """Load disease detection model, downloading via huggingface_hub if missing or an LFS pointer."""
    # Try downloading using huggingface_hub first for container environment robustness
    try:
        from huggingface_hub import hf_hub_download
        # We download models/export/model.h5
        path = hf_hub_download(
            repo_id="dhina4213/cropshield-ai",
            filename="models/export/model.h5",
            repo_type="space"
        )
        if os.path.exists(path) and os.path.getsize(path) > 1000000:
            model = tf.keras.models.load_model(path)
            return model, path
    except Exception as e:
        st.warning(f"Could not download model from Hugging Face Space: {e}. Trying local fallback...")

    potential_paths = [
        "models/export/model.h5",
        "models/mobilenetv2_final_best.h5",
        "models/export/best_model_phase2.h5",
        "models/export/best_model_phase1.h5",
        "model/plant_disease_model.h5"
    ]
    for path in potential_paths:
        if os.path.exists(path) and os.path.getsize(path) > 1000000:
            try:
                model = tf.keras.models.load_model(path)
                return model, path
            except Exception as e:
                continue
    # Fallback to loading the placeholder if absolutely nothing else exists
    try:
        model = tf.keras.models.load_model("model/plant_disease_model.h5")
        return model, "model/plant_disease_model.h5"
    except Exception as e:
        st.error(f"❌ No valid model checkpoints found: {e}")
        return None, None


@st.cache_resource
def load_class_indices(num_classes: int):
    """Load class name mappings matching the model's output size"""
    # 1. Try directory scan first (highly accurate for locally trained models)
    for path in ['data/PlantVillage', 'data/train', 'data/val']:
        if os.path.exists(path):
            dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            if len(dirs) == num_classes:
                return {str(i): name for i, name in enumerate(dirs)}

    # 2. Try loading from model/class_indices.json if the length matches
    try:
        with open("model/class_indices.json") as f:
            indices_dict = json.load(f)
            if len(indices_dict) == num_classes:
                return indices_dict
    except:
        pass

    # 3. Fallback to standard classes
    from utils.disease_info import DISEASE_INFO
    standard_classes = list(DISEASE_INFO.keys())
    if num_classes == 38:
        return {str(i): name for i, name in enumerate(standard_classes)}

    # 4. Try scanning directories and take a subset if count is greater/different
    for path in ['data/PlantVillage', 'data/train', 'data/val']:
        if os.path.exists(path):
            dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
            if len(dirs) >= num_classes:
                return {str(i): name for i, name in enumerate(dirs[:num_classes])}

    # 5. Final fallback
    return {str(i): standard_classes[i] if i < len(standard_classes) else f"Class {i}" for i in range(num_classes)}


@st.cache_resource
def load_llm():
    """Load LLM client - cached so it loads once"""
    try:
        client = get_llm_client()
        return client
    except Exception as e:
        st.warning(f"⚠️ LLM not available: {e}")
        return None


def get_severity(class_name: str) -> str:
    """Get disease severity level"""
    if "healthy" in class_name.lower():
        return "None"
    critical = ["late_blight", "yellow_leaf_curl"]
    high     = ["early_blight", "black_rot", "mosaic"]
    if any(c in class_name.lower() for c in critical):
        return "Critical"
    if any(h in class_name.lower() for h in high):
        return "High"
    return "Medium"


# ─── SESSION STATE INIT ───────────────────────
if "chat_history"    not in st.session_state:
    st.session_state.chat_history    = []
if "disease_context" not in st.session_state:
    st.session_state.disease_context = {}
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# Load resources at startup to populate sidebar correctly
model_res = load_model()
model, model_path = model_res if model_res else (None, None)

if model:
    num_classes = model.output_shape[-1]
    class_indices = load_class_indices(num_classes)
else:
    class_indices = None

llm = load_llm()

# Initialize LLM advisor
advisor      = None
chat_handler = None
if llm:
    advisor      = PlantDiseaseAdvisor(llm)
    chat_handler = ChatHandler(llm)


# ─── SIDEBAR ─────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 PlantDoc AI")
    st.markdown("*AI-Powered Plant Disease Expert*")
    st.markdown("---")

    # LLM Status
    st.markdown("### 🤖 AI Status")
    if llm:
        provider = os.getenv("DEFAULT_LLM", "groq")
        st.success(f"✅ AI Online ({provider})")
    else:
        st.error("❌ AI Offline")
        st.info("Add API key to .env file")

    language = "English"

    # Model info
    st.markdown("### 📊 Model Info")
    if model:
        model_filename = os.path.basename(model_path) if model_path else "Unknown"
        model_arch = "MobileNetV2"
        model_path_lower = model_path.lower() if model_path else ""
        if "efficientnet" in model_path_lower:
            model_arch = "EfficientNet"
        elif "resnet" in model_path_lower:
            model_arch = "ResNet50"
        else:
            for layer in model.layers:
                if "efficientnet" in layer.name.lower():
                    model_arch = "EfficientNet"
                    break
                elif "resnet" in layer.name.lower():
                    model_arch = "ResNet"
                    break
        st.markdown(f"""
        - **Loaded Model**: `{model_filename}`
        - **Architecture**: {model_arch}
        - **Classes**: {num_classes} categories
        - **Inference**: TensorFlow
        """)
    else:
        st.markdown("""
        - **Architecture**: None
        - **Status**: Offline
        """)

    st.markdown("---")
    st.markdown("### 🌱 Supported Plants")
    plants = [
        "🍎 Apple", "🍅 Tomato", "🥔 Potato",
        "🍇 Grape", "🌽 Corn", "🍑 Peach",
        "🫑 Pepper", "🍓 Strawberry", "🫐 Blueberry"
    ]
    for p in plants:
        st.markdown(f"  {p}")


# ─── MAIN PAGE ───────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌿 PlantDoc AI</h1>
    <p>Upload a plant leaf image for instant AI diagnosis
    + expert treatment advice</p>
</div>
""", unsafe_allow_html=True)

if not model or not class_indices:
    st.error("❌ Model files not found!")
    st.info("Place model files in model/ folder")
    st.stop()


# ─── IMAGE UPLOAD SECTION ────────────────────
st.markdown("## 📤 Upload Plant Leaf Image")

upload_col, camera_col = st.columns(2)

with upload_col:
    uploaded_file = st.file_uploader(
        "Choose image file",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Upload a clear image of the plant leaf"
    )

with camera_col:
    camera_photo = st.camera_input(
        "Or take a photo"
    )

# Get image
image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
elif camera_photo:
    image = Image.open(camera_photo).convert('RGB')


# ─── PREDICTION + AI ANALYSIS ────────────────
if image is not None:

    img_col, result_col = st.columns([1, 1])

    with img_col:
        st.markdown("### 📸 Specimen Visualization")
        img_tabs = st.tabs(["Original Photo", "👁️ AI Attention Map (Grad-CAM)"])
        
        with img_tabs[0]:
            st.image(image, use_container_width=True)
            w, h = image.size
            st.caption(f"Size: {w}×{h} pixels")
            
        with img_tabs[1]:
            # Generate and show Grad-CAM attention heatmap overlay
            with st.spinner("👁️ Generating Grad-CAM Attention Heatmap..."):
                from utils.gradcam import create_gradcam_comparison
                gradcam_res = create_gradcam_comparison(model, image, img_size=224)
                if gradcam_res["success"]:
                    st.image(gradcam_res["overlay"], use_container_width=True)
                    st.caption("The colored areas indicate where the deep learning network focused its attention to make the diagnosis (Red/Yellow = High Focus, Blue = Low Focus).")
                else:
                    st.error(f"Could not generate attention map: {gradcam_res['error_message']}")

    with result_col:
        st.markdown("### 🔬 AI Diagnosis")

        # Run prediction
        with st.spinner("🔍 Analyzing image and validating quality..."):
            try:
                from utils.preprocess import preprocess_image
                batch_img, original_resized = preprocess_image(image, validate_leaf=True)
                
                # Perform prediction using processed image
                predictions = model.predict(batch_img, verbose=0)
                pred_idx    = np.argmax(predictions[0])
                confidence  = float(predictions[0][pred_idx]) * 100
                class_name  = class_indices.get(str(pred_idx), "Unknown")

                # Top 5
                top5_idx = np.argsort(predictions[0])[-5:][::-1]
                top5 = [
                    {
                        "class"     : class_indices.get(str(i), "Unknown"),
                        "confidence": float(predictions[0][i]) * 100
                    }
                    for i in top5_idx
                ]
            except ValueError as e:
                err_msg = str(e)
                if "Leaf" in err_msg:
                    st.error("❌ Invalid Specimen Image!")
                else:
                    st.error("❌ Image Quality Check Failed!")
                st.warning(err_msg)
                st.info("💡 Tip: Ensure your photo is sharp, well-lit, and contains a clear view of a plant leaf.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
                st.stop()

        # Parse class name
        parts      = class_name.split("___")
        plant_name = parts[0].replace("_", " ")
        disease    = (parts[1].replace("_", " ")
                      if len(parts) > 1 else "Unknown")
        severity   = get_severity(class_name)
        is_healthy = "healthy" in class_name.lower()

        # Save to session state
        st.session_state.disease_context = {
            "plant"     : plant_name,
            "disease"   : disease,
            "class_name": class_name,
            "confidence": confidence,
            "severity"  : severity,
            "is_healthy": is_healthy
        }
        st.session_state.prediction_done = True

        # Show result
        if is_healthy:
            st.success("✅ Plant is HEALTHY!")
            st.balloons()
        else:
            st.error("⚠️ Disease Detected!")

        # Prediction card
        st.markdown(f"""
        <div class="pred-card">
            <h2>🌿 {plant_name}</h2>
            <h3>{disease}</h3>
            <h1>{confidence:.1f}%</h1>
            <p>Confidence Score</p>
        </div>
        """, unsafe_allow_html=True)

        # Severity badge
        badge_class = f"badge-{severity.lower()}"
        st.markdown(
            f'<p style="text-align:center;margin-top:10px">'
            f'Severity: <span class="{badge_class}">'
            f'{severity}</span></p>',
            unsafe_allow_html=True
        )

        # Top 5 predictions
        with st.expander("📊 Top 5 Predictions"):
            for pred in top5:
                name  = pred["class"].replace(
                    "___", " - ").replace("_", " ")
                conf  = pred["confidence"]
                color = ("green"
                         if "healthy" in pred["class"].lower()
                         else "red")
                st.markdown(
                    f'<span style="color:{color}">'
                    f'{name}</span>',
                    unsafe_allow_html=True
                )
                st.progress(conf / 100)
                st.caption(f"{conf:.1f}%")

    # ─── AI ANALYSIS TABS ────────────────────
    if llm and not is_healthy:

        st.markdown("---")
        st.markdown("## 🤖 AI Expert Analysis")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Disease Info",
            "💊 Treatment Plan",
            "🛡️ Prevention",
            "📄 Full Report",
            "💬 Ask AI"
        ])

        # TAB 1: Disease Info
        with tab1:
            with st.spinner(
                "🤖 AI is analyzing the disease..."
            ):
                explanation = advisor.explain_disease(
                    disease_name=disease,
                    plant_name=plant_name,
                    confidence=confidence
                )

            st.markdown(
                f'<div class="ai-response">'
                f'{explanation}</div>',
                unsafe_allow_html=True
            )

            # Translate if needed
            if language != "English":
                if st.button(f"🌍 Translate to {language}"):
                    with st.spinner("Translating..."):
                        translated = advisor.translate_advice(
                            explanation, language
                        )
                    st.markdown(
                        f'<div class="ai-response">'
                        f'{translated}</div>',
                        unsafe_allow_html=True
                    )

        # TAB 2: Treatment Plan
        with tab2:
            with st.spinner(
                "💊 Creating treatment plan..."
            ):
                treatment = advisor.get_treatment_plan(
                    disease_name=disease,
                    plant_name=plant_name,
                    severity=severity
                )

            st.markdown(
                f'<div class="ai-response">'
                f'{treatment}</div>',
                unsafe_allow_html=True
            )

        # TAB 3: Prevention
        with tab3:
            with st.spinner(
                "🛡️ Getting prevention tips..."
            ):
                prevention = advisor.get_prevention_tips(
                    disease_name=disease,
                    plant_name=plant_name
                )

            st.markdown(
                f'<div class="ai-response">'
                f'{prevention}</div>',
                unsafe_allow_html=True
            )

        # TAB 4: Full Report
        with tab4:
            st.markdown(
                "Generate a complete professional report"
            )

            if st.button(
                "📄 Generate Full Report",
                type="primary",
                use_container_width=True
            ):
                with st.spinner(
                    "📄 Generating report..."
                ):
                    report = advisor.generate_full_report(
                        disease_name=disease,
                        plant_name=plant_name,
                        confidence=confidence,
                        severity=severity
                    )

                st.markdown(
                    f'<div class="ai-response">'
                    f'{report}</div>',
                    unsafe_allow_html=True
                )

                # Download button
                st.download_button(
                    label="⬇️ Download Report (.txt)",
                    data=report,
                    file_name=(
                        f"{plant_name}_{disease}_report.txt"
                    ),
                    mime="text/plain",
                    use_container_width=True
                )

        # TAB 5: Chat with AI
        with tab5:
            st.markdown("### 💬 Chat with PlantDoc AI")
            st.markdown(
                "*Ask any question about your plant disease*"
            )

            # Set context for chat
            chat_handler.set_disease_context(
                plant=plant_name,
                disease=disease,
                confidence=confidence,
                severity=severity
            )

            # Suggested questions
            st.markdown("**Quick Questions:**")
            suggestions = (
                chat_handler.get_suggested_questions()
            )

            cols = st.columns(2)
            for i, q in enumerate(suggestions[:4]):
                col = cols[i % 2]
                with col:
                    if st.button(q, key=f"q_{i}",
                                 use_container_width=True):
                        st.session_state.chat_history.append(
                            {"role": "user", "content": q}
                        )
                        with st.spinner("🤖 Thinking..."):
                            response = chat_handler.chat(q)
                        st.session_state.chat_history.append(
                            {"role": "assistant",
                             "content": response}
                        )

            # Display chat history
            st.markdown("---")
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(
                            f'<div class="chat-user">'
                            f'👨🌾 {msg["content"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="chat-ai">'
                            f'🤖 {msg["content"]}</div>',
                            unsafe_allow_html=True
                        )

            # Chat input
            user_input = st.chat_input(
                "Ask about your plant disease..."
            )

            if user_input:
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input}
                )
                with st.spinner("🤖 Thinking..."):
                    response = chat_handler.chat(user_input)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
                st.rerun()

            # Clear chat button
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                chat_handler.clear_history()
                st.rerun()

    # Healthy plant message
    elif is_healthy:
        st.success("""
        ✅ Great news! Your plant appears healthy!
        
        Keep up these good practices:
        - Regular watering schedule
        - Proper sunlight exposure
        - Monitor for early disease signs
        - Maintain soil health
        """)

else:
    # Landing page when no image
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        ### 🔬 How It Works
        1. Upload leaf photo
        2. AI detects disease
        3. Get expert advice
        4. Chat with AI doctor
        """)

    with c2:
        st.markdown("""
        ### 🤖 AI Features
        - Disease explanation
        - Treatment plans
        - Prevention tips
        - Multi-language support
        - Chat with AI expert
        """)

    with c3:
        st.markdown("""
        ### 📊 Model Stats
        - 38 disease classes
        - 97%+ accuracy
        - 14 plant types
        - Instant results
        - Free to use
        """)

# ─── FOOTER ──────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8rem">
    🌿 PlantDoc AI | Built with TensorFlow + Groq LLaMA3
    <br>
    ⚠️ For educational use. Consult agricultural expert
    for professional diagnosis.
</div>
""", unsafe_allow_html=True)
