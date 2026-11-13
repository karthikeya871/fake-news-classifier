---
title: Fake News Classifier AI
emoji: 📰
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI Fake News Classifier & Explainable AI (XAI)

A production-ready NLP application that predicts whether a given news article is Real or Fake using Calibrated Linear SVM and Multinomial Naïve Bayes, featuring Explainable AI (XAI) keyword contribution scoring.

Optimized for ultra-low memory usage (< 45 MB RAM) to seamlessly run on free cloud hosting platforms without triggering out-of-memory (OOM) limits.

---

## 🚀 Free Deployment Guide

### Option 1: Hugging Face Spaces (Recommended - 16 GB Free RAM)
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set **Space name** (e.g., `fake-news-classifier`).
3. Select **Docker** as the Space SDK.
4. Choose **Public** or **Private**, and click **Create Space**.
5. Connect your GitHub repository (`karthikeya871/fake-news-classifier`) or push code directly to the Hugging Face git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/<SPACE_NAME>
   git push space main
   ```
6. Hugging Face will automatically build using `Dockerfile` and expose the app on port `7860`.

---

### Option 2: Render (Free Web Service - 512 MB RAM)
1. Go to [Render Dashboard](https://dashboard.render.com/) and click **New +** -> **Web Service**.
2. Connect your GitHub repository (`karthikeya871/fake-news-classifier`).
3. If using Render Blueprints, Render will automatically detect `render.yaml`.
4. Alternatively, configure manually:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --worker-class gthread --timeout 120 app:app`
   - **Instance Type**: `Free (512 MB)`
5. Click **Deploy Web Service**. Because the cleaned models use under 45 MB of RAM, it deploys instantly with zero OOM errors!

---

### Option 3: Railway / Koyeb / Fly.io
Both `Dockerfile` and `Procfile` are included in the repository root and inner directories:
- **Port**: Supports `${PORT:-7860}` dynamically.
- **Workers**: Uses 1 gunicorn worker with 2 lightweight gthread threads to maximize concurrency while maintaining a minimal RAM footprint.

---

## 🛠 Local Development & Testing

1. **Activate virtual environment**:
   ```bash
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python app.py
   ```
4. Open [http://localhost:7860](http://localhost:7860) in your browser.
