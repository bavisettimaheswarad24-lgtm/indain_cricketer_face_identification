import os
import io
import json
import traceback
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization

# -------------------------------------------------------------------------
# Page Configuration & Styling
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Cricketers Face Identification",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Lord's Claret & Gold Cricket Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp {
        background: radial-gradient(circle at 50% 0%, #152445 0%, #070c18 60%, #03060d 100%);
        color: #f1f5f9;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .lords-top-ribbon {
        height: 6px;
        background: linear-gradient(90deg, #9e1b32 0%, #c5a059 50%, #9e1b32 100%);
        margin-bottom: 1.5rem;
        border-radius: 3px;
    }

    .main-title-container {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(197, 160, 89, 0.35);
        margin-bottom: 2rem;
    }

    .sub-badge {
        display: inline-block;
        background: rgba(158, 27, 50, 0.35);
        color: #e6ca85;
        border: 1px solid rgba(197, 160, 89, 0.5);
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    .app-main-heading {
        font-family: 'Playfair Display', serif;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #ffffff;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.7);
    }

    .app-tagline {
        color: #cbd5e1;
        font-size: 0.98rem;
        margin-top: 0.5rem;
    }

    /* Card Styling */
    .player-card {
        background: linear-gradient(135deg, rgba(13, 22, 41, 0.95) 0%, rgba(8, 14, 27, 0.98) 100%);
        border: 1px solid rgba(197, 160, 89, 0.4);
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5);
    }

    .player-name-banner {
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .player-role-badge {
        display: inline-block;
        background: linear-gradient(135deg, #9e1b32 0%, #6e1323 100%);
        color: #ffffff;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .match-confidence {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d27a;
    }

    .stat-pill {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(197, 160, 89, 0.25);
        border-radius: 8px;
        padding: 0.85rem;
        margin-bottom: 0.6rem;
    }

    .stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #e6ca85;
        font-weight: 700;
    }

    .stat-value {
        font-size: 1rem;
        color: #ffffff;
        font-weight: 600;
        margin-top: 3px;
    }
</style>
<div class="lords-top-ribbon"></div>
<div class="main-title-container">
    <div class="sub-badge">AI Biometric Studio • 324 Indian Cricketers</div>
    <h1 class="app-main-heading">INDIAN CRICKETERS FACE IDENTIFICATION</h1>
    <p class="app-tagline">Deep Metric Recognition Powered by FaceNet (InceptionResnetV1) &amp; MTCNN</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# Resource Loading
# -------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATABASE_PATH = os.path.join(DATA_DIR, "player_database.json")
NPZ_PATH = os.path.join(MODELS_DIR, "cricket_face_embeddings.npz")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner="Initializing MTCNN detector & FaceNet neural network...")
def get_models():
    detector = MTCNN(
        keep_all=True,
        select_largest=False,
        post_process=True,
        min_face_size=30,
        device=DEVICE
    )
    embedder = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    return detector, embedder

@st.cache_data(show_spinner="Loading 324 cricketers database & vectors...")
def get_data():
    database = {}
    if os.path.exists(DATABASE_PATH):
        with open(DATABASE_PATH, "r", encoding="utf-8") as f:
            database = json.load(f)

    centroids = None
    exemplars = None
    labels = None
    players = []
    exemplar_indices = []

    if os.path.exists(NPZ_PATH):
        npz = np.load(NPZ_PATH, allow_pickle=True)
        centroids = npz['centroids']
        exemplars = npz.get('exemplars', None)
        labels = npz.get('labels', None)
        players = list(npz['players'])
        if exemplars is not None and labels is not None:
            exemplar_indices = [np.where(labels == i)[0] for i in range(len(players))]

    return database, centroids, exemplars, labels, players, exemplar_indices

try:
    detector, embedder = get_models()
    database, centroids, exemplars, labels, players, exemplar_indices = get_data()
    loaded_ok = True
except Exception as e:
    st.error(f"Error during initialization: {str(e)}")
    st.code(traceback.format_exc())
    loaded_ok = False

# -------------------------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    min_confidence = st.slider("Strict Identification Threshold", 0.40, 0.90, 0.55, 0.05)
    st.markdown(f"**Database:** `{len(players)}` Cricketers Indexed")
    st.markdown(f"**Compute Device:** `{DEVICE.upper()}`")
    
    st.markdown("---")
    st.markdown("### ⭐ Quick Test Stars")
    sample_dir = os.path.join(BASE_DIR, "static", "samples")
    sample_options = ["None (Upload Your Own)"]
    if os.path.exists(sample_dir):
        sample_files = [f for f in sorted(os.listdir(sample_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        sample_options.extend(sample_files)
    
    selected_sample = st.selectbox("Select Star Sample Photo", sample_options)

    st.markdown("---")
    with st.expander("🏛️ View All 324 Cricketers Roster"):
        search_query = st.text_input("Search cricketer", "")
        filtered = [p for p in players if search_query.lower() in p.lower()]
        st.write(f"Showing {len(filtered)} players:")
        st.dataframe(filtered, use_container_width=True, height=250)

# -------------------------------------------------------------------------
# Image Input & Biometric Processing
# -------------------------------------------------------------------------
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 📸 Input Photograph")
    input_mode = st.radio("Choose Input Method", ["File Upload", "Live Camera", "Quick Sample Star"], horizontal=True)
    
    input_image = None

    if input_mode == "Quick Sample Star" and selected_sample != "None (Upload Your Own)":
        sample_path = os.path.join(sample_dir, selected_sample)
        if os.path.exists(sample_path):
            input_image = Image.open(sample_path).convert("RGB")
    elif input_mode == "Live Camera":
        cam_pic = st.camera_input("Capture Face from Webcam")
        if cam_pic:
            input_image = Image.open(cam_pic).convert("RGB")
    else:
        uploaded_file = st.file_uploader("Upload Cricket Player Photo (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file:
            input_image = Image.open(uploaded_file).convert("RGB")

def match_face(embedding):
    """
    Balanced Centroid + Top Exemplar scoring (50% Centroid + 35% Top-1 + 15% Top-2)
    Prevents single-outlier false positives.
    """
    centroid_sims = np.dot(centroids, embedding)
    
    if exemplars is not None and len(exemplar_indices) == len(players):
        all_ex_sims = np.dot(exemplars, embedding)
        top1_sims = np.zeros(len(players), dtype=np.float32)
        top2_sims = np.zeros(len(players), dtype=np.float32)
        
        for idx, ex_idx in enumerate(exemplar_indices):
            if len(ex_idx) == 0:
                top1_sims[idx] = centroid_sims[idx]
                top2_sims[idx] = centroid_sims[idx]
            elif len(ex_idx) == 1:
                top1_sims[idx] = all_ex_sims[ex_idx[0]]
                top2_sims[idx] = all_ex_sims[ex_idx[0]]
            else:
                p_sims = np.sort(all_ex_sims[ex_idx])
                top1_sims[idx] = p_sims[-1]
                top2_sims[idx] = 0.5 * (p_sims[-1] + p_sims[-2])
                
        final_scores = 0.50 * centroid_sims + 0.35 * top1_sims + 0.15 * top2_sims
    else:
        final_scores = centroid_sims
        
    best_idx = int(np.argmax(final_scores))
    best_score = float(final_scores[best_idx])
    return best_idx, best_score

if loaded_ok and input_image is not None:
    # Downscale large images for sub-second MTCNN detection
    orig_w, orig_h = input_image.size
    max_dim = 900
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        proc_img = input_image.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.BILINEAR)
    else:
        proc_img = input_image

    with st.spinner("Extracting facial biometric features & searching 324 legends..."):
        boxes, probs = detector.detect(proc_img)
        
        detected_faces = []
        annotated_image = proc_img.copy()
        draw = ImageDraw.Draw(annotated_image)

        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                prob = probs[i] if probs is not None else 1.0
                if prob < 0.70:
                    continue
                
                face_tensor = extract_face(proc_img, box, image_size=160, margin=14)
                if face_tensor is None:
                    continue
                
                face_tensor = fixed_image_standardization(face_tensor)
                face_tensor = face_tensor.unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    raw_emb = embedder(face_tensor).cpu().numpy()[0]
                    norm = np.linalg.norm(raw_emb)
                    emb = raw_emb / (norm + 1e-9)
                    
                match_idx, score = match_face(emb)
                player_raw_name = players[match_idx]
                
                clean_key = player_raw_name.lower().replace(' ', '_')
                info = database.get(clean_key, {})
                display_name = info.get("display_name", player_raw_name.replace('_', ' ').title())

                detected_faces.append({
                    "box": box,
                    "prob": prob,
                    "score": score,
                    "player_key": clean_key,
                    "display_name": display_name,
                    "info": info
                })

                x1, y1, x2, y2 = [int(v) for v in box]
                color = "#00d27a" if score >= min_confidence else "#eab308"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                label_text = f"{display_name} ({score * 100:.1f}%)"
                draw.text((x1 + 4, max(4, y1 - 22)), label_text, fill=color)

    with col_input:
        st.image(annotated_image, caption=f"Analysis Complete: {len(detected_faces)} Face(s) Detected", use_container_width=True)

    with col_results:
        st.markdown("### 🏆 Identification Dossier")
        
        if len(detected_faces) == 0:
            st.warning("⚠️ No human face detected in this photograph. Please upload a clear photo with visible facial features.")
        else:
            for idx, item in enumerate(detected_faces):
                score = item["score"]
                info = item["info"]
                d_name = item["display_name"]
                
                if score < min_confidence:
                    st.error(f"⚠️ Face #{idx+1}: Low confidence match ({score*100:.1f}%). This cricketer may not be in the database.")
                
                st.markdown(f"""
                <div class="player-card">
                    <div class="player-role-badge">{info.get("role", "INDIAN CRICKETER")}</div>
                    <div class="player-name-banner">{d_name}</div>
                    <p style="color: #cbd5e1; font-size: 0.9rem; margin-bottom: 0.8rem;">
                        <strong>Active Era:</strong> {info.get("active_era", "International & Domestic Circuit")}
                    </p>
                    <div style="margin: 0.8rem 0;">
                        <span style="color: #cbd5e1; font-size: 0.9rem;">Neural Metric Confidence: </span>
                        <span class="match-confidence">{score * 100:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="stat-pill">
                        <div class="stat-label">🏏 Batting Discipline</div>
                        <div class="stat-value">{info.get("batting_style", "Right-hand bat")}</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-label">🎯 Bowling Specialization</div>
                        <div class="stat-value">{info.get("bowling_style", "Right-arm medium")}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="stat-pill">
                        <div class="stat-label">📊 Career Matches</div>
                        <div class="stat-value">{info.get("career_matches", "International & Domestic")}</div>
                    </div>
                    <div class="stat-pill">
                        <div class="stat-label">🏟️ Representative Teams</div>
                        <div class="stat-value">{info.get("teams", "India National Cricket Team")}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("🌟 Historic Highlights & Performance", expanded=True):
                    st.markdown(f"**⭐ Best Historic Performance:** {info.get('best_performance', 'Match-winning performance')}")
                    highlights = info.get("highlights", [])
                    if isinstance(highlights, list) and len(highlights) > 0:
                        st.markdown("**🎖️ Honours & Milestones:**")
                        for h in highlights:
                            st.markdown(f"- {h}")
                            
                with st.expander("📖 Biography & Legacy", expanded=False):
                    st.write(info.get("biography", "A proud member of the Indian cricket fraternity."))
else:
    with col_results:
        st.info("👈 Upload a photograph, take a webcam photo, or select a Quick Test Star to begin face identification.")
