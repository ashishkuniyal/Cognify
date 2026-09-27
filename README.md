# 🎓 Intelligent EdTech Platform
### *Student Matrix V2 — Multi-Engine ML Architecture*

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61dafb?logo=react)](https://reactjs.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Multi--Output-orange)](https://xgboost.ai)
[![Gemini](https://img.shields.io/badge/Gemini_AI-3.8_Flash-purple?logo=google)](https://ai.google.dev)
[![Flask](https://img.shields.io/badge/Flask-REST_API-black?logo=flask)](https://flask.palletsprojects.com)

> An enterprise-grade EdTech ML platform that predicts full academic potential across Math, Reading, and Writing simultaneously, identifies at-risk students, recommends career paths via K-Means clustering, and provides personalized guidance through a Gemini AI Counselor with multi-turn memory.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Intelligent EdTech Platform               │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React + Vite)                                     │
│  ├── Prediction Engine   (Radar Chart · 3-Score Output)      │
│  ├── Live Analytics      (Area Chart · MLOps Drift Monitor)  │
│  └── AI Counselor        (Gemini 3.8 Flash · Multi-turn)     │
├─────────────────────────────────────────────────────────────┤
│  Backend (Flask REST API)                                    │
│  ├── /api/predict   → Multi-Output Regression + Prescriptive │
│  ├── /api/analytics → SQLite inference logs                  │
│  ├── /api/chat      → Gemini AI with session memory          │
│  └── /api/chat/reset → Clear conversation history           │
├─────────────────────────────────────────────────────────────┤
│  ML Engines (3 Parallel Models)                              │
│  ├── model_radar.pkl    → XGBoost MultiOutputRegressor       │
│  ├── model_atrisk.pkl   → Random Forest Classifier           │
│  └── model_cluster.pkl  → K-Means (3 Career Clusters)        │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-Target Regression** | Predicts Math, Reading & Writing scores simultaneously using XGBoost |
| **At-Risk Detection** | Random Forest Classifier flags students with high failure probability |
| **Career Clustering** | K-Means groups students into STEM, Humanities, or Business tracks |
| **AI Counselor** | Gemini 3.8 Flash chatbot with full multi-turn conversation memory |
| **Prescriptive Advisor** | What-If simulation showing score boost from completing test prep |
| **Live Analytics** | Real-time inference stream with data drift monitoring |
| **Model Fallback** | Cascade failover across 3 Gemini models on 503 errors |
| **Dark/Light Mode** | Full theme toggle with glassmorphism design |

---

## 🚀 Quick Start

### Backend
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# Train the ML models
python src/components/data_ingestion.py

# Run the API server
python app.py
```

### Frontend
```bash
cd frontend
npm install

# Set API URL (create frontend/.env)
echo "VITE_API_URL=http://localhost:5000" > .env

npm run dev
```

---

## 📁 Project Structure

```
intelligent-edtech-platform/
├── artifacts/                  # Serialized ML models (.pkl)
│   ├── model_radar.pkl         # Multi-Output XGBoost
│   ├── model_atrisk.pkl        # Random Forest Classifier
│   ├── model_cluster.pkl       # K-Means Clusterer
│   └── preprocessor.pkl        # Feature transformer
├── frontend/                   # React/Vite UI
│   ├── src/
│   │   ├── App.jsx             # Main application
│   │   └── index.css           # Design system
│   ├── .env                    # VITE_API_URL (local)
│   └── package.json
├── src/
│   ├── components/
│   │   ├── data_ingestion.py   # Training pipeline entry point
│   │   ├── data_transformation.py # Feature engineering
│   │   └── model_trainer.py    # 3-engine training
│   └── pipeline/
│       └── predict_pipeline.py # Inference orchestrator
├── app.py                      # Flask REST API
├── Procfile                    # Render deployment
├── requirements.txt
└── .env                        # GEMINI_API_KEY (backend)
```

---

## 🧠 ML Pipeline Details

### Training
The pipeline engineers `at_risk` (binary) and multi-target labels from the raw dataset, then trains three independent models:
1. **XGBoost MultiOutputRegressor** → Predicts [Math, Reading, Writing] in a single pass
2. **RandomForestClassifier** → Binary classification: at-risk (score < 50 threshold)
3. **KMeans (k=3)** → Clusters predicted score profiles into career tracks

### Inference
`PredictPipeline` loads all three artifacts and runs them sequentially on incoming demographic data. A hidden **What-If Engine** simulates the `test_preparation_course=completed` scenario to calculate a personalized prescriptive score boost.

---

## 🌐 Deployment

- **Backend:** [Render](https://render.com) (Free tier) — uses `gunicorn` via `Procfile`
- **Frontend:** [Vercel](https://vercel.com) (Free tier) — set `VITE_API_URL` env var to backend URL

See [deployment_guide.md](./deployment_guide.md) for full step-by-step instructions.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| ML | XGBoost, scikit-learn, pandas, numpy |
| Backend | Python 3.11, Flask, SQLAlchemy, Flask-Session |
| AI | Google Gemini 3.8 Flash (`google-genai`) |
| Frontend | React 19, Vite, Recharts, Framer Motion, Lucide |
| Deployment | Render (backend), Vercel (frontend) |
| Tracking | MLflow (local) |

---

*Built as a portfolio project demonstrating full-stack ML engineering, from feature engineering to cloud deployment.*