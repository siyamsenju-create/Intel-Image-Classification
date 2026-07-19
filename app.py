import os
import random
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from src.predict import predict_image

# ----------------------------------------------------
# 1. Page Configuration & Custom CSS
# ----------------------------------------------------
st.set_page_config(
    page_title="Intel Scene Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    /* Light background theme */
    .stApp {
        background-color: #f0f4f8;
        color: #1e293b;
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Card containers */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.07);
    }
    
    /* Metric container */
    .metric-box {
        text-align: center;
        padding: 16px;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-radius: 12px;
        border: 1px solid #bfdbfe;
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1d4ed8;
    }
    
    /* Prediction success box */
    .prediction-title {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 500;
    }
    .prediction-value {
        font-size: 2rem;
        font-weight: 800;
        color: #059669;
        margin-bottom: 12px;
    }
    
    /* Sidebar custom styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0f2744 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h2 { color: #7dd3fc !important; }
    
    /* Custom progress bar wrapper */
    .custom-bar-container {
        margin-bottom: 12px;
    }
    .custom-bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #475569;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .custom-bar-bg {
        background-color: #e2e8f0;
        border-radius: 8px;
        height: 14px;
        overflow: hidden;
    }
    .custom-bar-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
    }
    div[data-testid="stFileUploaderDropzone"] {
        background-color: #f8fafc !important;
        border: 2px dashed #93c5fd !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. Path Setup & Utility Functions
# ----------------------------------------------------
MODEL_PATH = "models/best_model.keras"
CLASS_NAMES_PATH = "results/class_names.json"
METRICS_PATH = "results/metrics.json"
CURVES_IMAGE_PATH = "results/training_curves.png"
CM_IMAGE_PATH = "results/confusion_matrix.png"
PRED_DIR = "data/pred"

@st.cache_resource
def load_classification_model():
    """Load model and cache it to avoid reloading on each run."""
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

def load_class_names():
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r') as f:
            return json.load(f)
    # Default fallbacks if training hasn't run yet
    return ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# ----------------------------------------------------
# 3. Sidebar Setup
# ----------------------------------------------------
st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>🖼️ Intel Scene Classifier</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Transfer Learning on MobileNetV2</p>", unsafe_allow_html=True)
st.sidebar.write("---")

menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard & Predictor", "📊 Model Insights", "ℹ️ Project Info"],
    index=0
)

# Load global resources
model = load_classification_model()
class_names = load_class_names()

# Check model availability and show alert in sidebar
if model is None:
    st.sidebar.warning("⚠️ Model file not found. Please train the model using 'python src/train.py' first.")

# ----------------------------------------------------
# 4. View Controllers
# ----------------------------------------------------

# View 1: Dashboard & Predictor
if menu == "🏠 Dashboard & Predictor":
    st.markdown("<h1>Scene Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-top: -15px;'>Upload an image or pick a random scene from the prediction set to classify it.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Image Input Selection")
        
        input_type = st.radio("Choose input method:", ["Upload Custom Image", "Select Random Sample from 'data/pred'"])
        
        selected_img_path = None
        uploaded_file = None
        
        if input_type == "Upload Custom Image":
            uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
        else:
            if not os.path.exists(PRED_DIR) or len(os.listdir(PRED_DIR)) == 0:
                st.info("The prediction folder 'data/pred' is empty or doesn't exist.")
            else:
                pred_images = [f for f in os.listdir(PRED_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Use session state to persist random choice across runs unless button clicked
                if 'random_img' not in st.session_state or st.button("🎲 Choose Another Random Image"):
                    st.session_state.random_img = random.choice(pred_images)
                    
                selected_img_path = os.path.join(PRED_DIR, st.session_state.random_img)
                st.success(f"Selected: `{st.session_state.random_img}`")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display Input Image
        img_to_predict = None
        if uploaded_file is not None:
            img_to_predict = Image.open(uploaded_file)
        elif selected_img_path is not None:
            img_to_predict = Image.open(selected_img_path)
            
        if img_to_predict is not None:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Selected Image Preview")
            st.image(img_to_predict, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
    with col2:
        st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
        st.subheader("Model Classification Analysis")
        
        if img_to_predict is None:
            st.info("Please upload an image or choose a random sample to run prediction.")
        elif model is None:
            st.error("Cannot run prediction because the model is not trained yet.")
        else:
            # We must save the image to a temporary file for the predict function
            temp_path = "temp_prediction_image.jpg"
            img_to_predict.convert("RGB").save(temp_path)
            
            with st.spinner("Analyzing image features..."):
                pred_class, confidence, prob_dict = predict_image(temp_path, model, class_names)
                
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            # Display primary prediction
            st.markdown(f"<div class='prediction-title'>Predicted Category</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-value'>{pred_class.upper()} ({confidence:.2%})</div>", unsafe_allow_html=True)
            
            st.write("---")
            st.subheader("Confidence Scores Breakdown")
            
            # Display custom progress bars for class probabilities
            for cls, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
                percentage = prob * 100
                st.markdown(f"""
                <div class="custom-bar-container">
                    <div class="custom-bar-label">
                        <span>{cls.capitalize()}</span>
                        <span>{prob:.2%}</span>
                    </div>
                    <div class="custom-bar-bg">
                        <div class="custom-bar-fill" style="width: {percentage}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        st.markdown("</div>", unsafe_allow_html=True)

# View 2: Model Insights
elif menu == "📊 Model Insights":
    st.markdown("<h1>Model Insights & Performance Metrics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; margin-top: -15px;'>Analyze training history, precision/recall reports, and category confusion matrices.</p>", unsafe_allow_html=True)
    
    # Check if metrics exist
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
            
    if metrics is not None:
        # High level summary cards
        st.subheader("Test Dataset Evaluation Summaries")
        col_acc, col_loss = st.columns(2)
        
        with col_acc:
            st.markdown(f"""
            <div class="card metric-box">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-value">{metrics['test_accuracy']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_loss:
            st.markdown(f"""
            <div class="card metric-box">
                <div class="metric-title">Test Categorical Loss</div>
                <div class="metric-value">{metrics['test_loss']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("---")
        
    # Plots column
    col_plots, col_table = st.columns([1, 1.2], gap="large")
    
    with col_plots:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Training History Curves")
        if os.path.exists(CURVES_IMAGE_PATH):
            st.image(CURVES_IMAGE_PATH, use_container_width=True, caption="Training & Validation Accuracy and Loss")
        else:
            st.info("Training history curves plot not found. Run training first.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Evaluation Confusion Matrix")
        if os.path.exists(CM_IMAGE_PATH):
            st.image(CM_IMAGE_PATH, use_container_width=True, caption="Confusion Matrix on Test Dataset")
        else:
            st.info("Confusion matrix plot not found. Run evaluation first.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_table:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Detailed Classification Report")
        
        if metrics is not None and 'classification_report' in metrics:
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            
            # Format report table beautifully
            # Filter class rows vs macro/weighted averages
            classes_only = report_df.loc[class_names]
            averages_only = report_df.loc[['accuracy', 'macro avg', 'weighted avg']]
            
            st.write("**Category Scores**")
            st.dataframe(
                classes_only[['precision', 'recall', 'f1-score', 'support']].style.format({
                    'precision': '{:.2%}',
                    'recall': '{:.2%}',
                    'f1-score': '{:.2%}',
                    'support': '{:.0f}'
                }),
                use_container_width=True
            )
            
            st.write("**Model Summaries**")
            st.dataframe(
                averages_only[['precision', 'recall', 'f1-score', 'support']].style.format({
                    'precision': '{:.2%}',
                    'recall': '{:.2%}',
                    'f1-score': '{:.2%}',
                    'support': '{:.0f}'
                }),
                use_container_width=True
            )
        else:
            st.info("Classification report not found. Run training and evaluation first.")
            
        st.markdown("</div>", unsafe_allow_html=True)

# View 3: Project Info
else:
    st.markdown("<h1>About the Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
        <h3>Dataset Information</h3>
        <p>The dataset used for this project is the <b>Intel Image Classification dataset</b>, originally provided by Intel for a data science challenge. It contains high-quality images (150x150 pixels) split into six distinct scenery categories:</p>
        <ul>
            <li>🌲 <b>Forest</b></li>
            <li>🏢 <b>Buildings</b></li>
            <li>🏔️ <b>Glacier</b></li>
            <li>⛰️ <b>Mountain</b></li>
            <li>🌊 <b>Sea</b></li>
            <li>🛣️ <b>Street</b></li>
        </ul>
        <p>This dataset features approximately 14,000 images for training, 3,000 for testing, and 7,300 for unlabelled predictions.</p>
    </div>
    
    <div class='card'>
        <h3>Deep Learning Methodology</h3>
        <p>To deliver a fast, efficient, and highly accurate classification system without requiring high-end GPU servers for training, we use <b>Transfer Learning with MobileNetV2</b>:</p>
        <ol>
            <li><b>Feature Extraction:</b> We leverage a MobileNetV2 base model pre-trained on ImageNet. Its weights are frozen, meaning it retains its powerful general feature representation abilities.</li>
            <li><b>Normalization:</b> Images are scaled to 150x150 RGB. Inputs are rescaled internally inside the model graph to the range [-1, 1] as expected by MobileNetV2.</li>
            <li><b>Custom Classification Head:</b> We append a Global Average Pooling layer, followed by a Dropout layer (rate=0.3) to prevent overfitting, and a final Dense softmax layer with 6 output units.</li>
            <li><b>Optimizer & Loss:</b> The model is trained using the Adam optimizer with categorical crossentropy loss.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
