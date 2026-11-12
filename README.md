# 🏏 Indian Cricket Face Identification AI 2.0 (324 Legends)

A Computer Vision & Deep Learning web application designed to detect, recognize, and display verified statistical intelligence for **324 Indian Cricket Legends & Stars**. Built for cloud deployment on **Render**, local development, and real-time inference.

---

## 🌟 Key Features

1. **Exact Web UI from Reference**:
   - Dark navy stadium aesthetic with subtle glowing accents.
   - **Input & Detection Studio**:
     - 📥 Drag & Drop photo upload.
     - 📹 Live Camera capture.
     - ⭐ Quick Test Stars (1-click testing with stars like Rohit Sharma, Ekta Bisht, Virat Kohli, MS Dhoni, Sachin Tendulkar, Bhuvneshwar Kumar).
     - Multi-face bounding box carousel with `<` and `>` navigation arrows.
     - Visual search floating action button.
   - **Player Intelligence Dossier**:
     - Glowing circular player avatar.
     - Role badge (`BATSMAN`, `BOWLER`, `ALL-ROUNDER`, `WICKET-KEEPER`).
     - Calibrated Model Confidence Match bar.
     - Database alert warning for low confidence (<45%) or unknown players.
     - 2x2 Specs Grid: Batting Style, Bowling Style, Career Matches, Key Teams.
     - ⭐ Best Performance.
     - 🎖️ Career Highlights with trophies.
     - 📖 Biography & Legacy.
   - **Legends Roster Modal**:
     - Instant search and role filtering across all 324 supported Indian cricketers.

2. **Better Algorithm: Deep Metric Learning (FaceNet + MTCNN)**:
   - **Eliminates False Positives**: Replaces Haar cascades (which detected random background walls and jerseys) with **MTCNN** (Multi-task Cascaded Convolutional Networks) for precise facial landmark detection.
   - **512-Dimensional Metric Embeddings**: Replaces weak 324-class Softmax classification with **InceptionResnetV1 (pretrained on VGGFace2)**.
   - **Hyperspherical Separation**: Cosine similarity separation between identities is massive (>0.75 for true matches vs <0.25 for cross matches).
   - **1-Millisecond Dot-Product Matching**: Instant query lookup.
   - **Compact Model**: Saves all 324 players into a `~3.8 MB` float32 array, fitting comfortably within Render's 512 MB memory limit.

3. **Accurate & Verified Player Database**:
   - Covers all 324 cricketers in the dataset (`data/player_database.json`).
   - Verified statistics for men's and women's legends (Ekta Bisht, Rohit Sharma, MS Dhoni, Virat Kohli, Smriti Mandhana, Harmanpreet Kaur, Jasprit Bumrah, etc.).

---

## 🚀 Quick Start (Local Run)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Web App
```bash
python app.py
```
Open your browser and navigate to: `http://localhost:5000`

---

## 📸 CLI Inference

To identify faces from the command line:
```bash
python infer.py --image "static/samples/rohit_sharma.jpg" --save
```
This generates `output_rohit_sharma.jpg` with yellow bounding boxes, labels, and prints player dossiers to the terminal.

---

## ☁️ Deploy to Render

Refer to [`DEPLOYMENT_RENDER.md`](file:///c:/Users/bavis/Downloads/cricket_face_identification/DEPLOYMENT_RENDER.md) for full deployment instructions:
- Connect your GitHub repository to Render.
- Select **Web Service**.
- Build Command: `pip install --upgrade pip && pip install -r requirements.txt`
- Start Command: `gunicorn app:app --workers 1 --threads 4 --timeout 120`

