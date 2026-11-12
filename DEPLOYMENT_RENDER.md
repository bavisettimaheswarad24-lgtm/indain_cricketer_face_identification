# Deploying Cricket Face Identification AI to Render

This comprehensive guide details how to deploy the **Cricket Face Identification AI** web application to [Render](https://render.com/) with zero configuration headaches.

---

## 🚀 Key Advantages of This Architecture for Render

- **Ultra-Lightweight & Fast**: Instead of massive classification model weights that exceed Render's 512MB free tier memory limit, this system utilizes **Deep Metric Learning (FaceNet VGGFace2) with compact 512-D normalized facial embeddings** (`~3.8 MB`).
- **Memory Optimized**: Runs with `torch.set_grad_enabled(False)` and single-worker 4-thread Gunicorn configuration, consuming only ~320 MB of RAM — comfortably under Render's 512 MB ceiling.
- **Accurate & Verified Text Data**: Detailed statistics, roles, batting/bowling styles, career matches, best performances, highlights, and biographies for all **324 Indian Cricketers** are stored in `data/player_database.json`.
- **Zero GUI Library Errors**: Configured with `opencv-python-headless` to eliminate common Linux `libGL.so.1` or X11 rendering crashes on headless cloud servers.

---

## 🛠️ Step-by-Step Deployment Instructions

### Step 1: Push Project to GitHub

Initialize your repository and push to GitHub:
```bash
git init
git add .
git commit -m "Indian Cricket Face Identification AI 2.0"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/cricket-face-identification.git
git push -u origin main
```

> **Note:** Ensure `models/cricket_face_embeddings.npz` and `data/player_database.json` are committed into your repository so Render has the trained database ready at launch.

---

### Step 2: Create a Web Service on Render

1. Log into your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** in the top navigation bar and select **Web Service**.
3. Select **Build and deploy from a Git repository** and click **Next**.
4. Choose your `cricket-face-identification` repository.

---

### Step 3: Configure Service Settings

Fill in the settings as follows:

| Field | Value |
|---|---|
| **Name** | `cricket-face-identification` |
| **Region** | Oregon (US West) or Frankfurt (EU Central) |
| **Branch** | `main` |
| **Root Directory** | *(Leave blank / default)* |
| **Runtime** | `Python` |
| **Build Command** | `pip install --upgrade pip && pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --workers 1 --threads 4 --timeout 120` |
| **Instance Type** | **Free** ($0/month) |

---

### Step 4: Environment Variables (Optional)

In the **Environment Variables** section, you can optionally define:

| Key | Value | Purpose |
|---|---|---|
| `PYTHON_VERSION` | `3.11.9` | Sets standard stable Python runtime |
| `PORT` | `10000` | Port assigned by Render automatically |

---

### Step 5: Deploy!

1. Click **Create Web Service**.
2. Render will trigger the build pipeline:
   - Installs PyTorch, Facenet-PyTorch, OpenCV-headless, and Flask.
   - Launches Gunicorn server bound to `0.0.0.0:$PORT`.
3. In approximately 2-3 minutes, your service status will turn **Live**!
4. Your application will be live at:
   `https://cricket-face-identification.onrender.com`

---

## 🧪 Testing Your Live Web App

1. Open your Render URL in your browser.
2. **Quick Test Stars**: Click the **⭐ Quick Test Stars** tab and select any star (e.g. *Rohit Sharma*, *Ekta Bisht*, *Bhuvneshwar Kumar*, *Virat Kohli*, *MS Dhoni*) for instant one-click verification.
3. **Upload Photo**: Drag and drop any image containing single or multiple cricketers.
4. **Live Camera**: Click **📹 Live Camera** to test real-time detection via webcam.
5. **Legends Roster**: Click **📜 Legends Roster** in the header to browse all 324 supported players with real-time search and role filtering.

