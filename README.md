<div align="center">
  <h1>🎓 Student Performance Prediction - Advanced ML Architecture</h1>
  <p>An End-to-End Machine Learning Pipeline with a Decoupled Full-Stack Web Application, MLOps, and Production Database Logging.</p>
</div>

---

## 🚀 Overview
This repository contains a complete, production-ready machine learning lifecycle project designed to predict student math scores based on demographic and historical data. It demonstrates modern ML Engineering and MLOps practices rather than just a basic Jupyter Notebook.

## 🏗️ Architecture & Tech Stack
*   **Machine Learning**: Scikit-Learn, Pandas, NumPy, XGBoost, CatBoost
*   **MLOps (Experiment Tracking)**: MLflow
*   **Backend API**: Python, Flask, Flask-RESTful, Flask-CORS
*   **Database**: SQLite & SQLAlchemy (Prediction Logging)
*   **Frontend**: React.js, Vite, Vanilla CSS (Glassmorphism design)
*   **Deployment**: Docker & Docker Compose

## ✨ Key Features
1.  **Fully Decoupled Architecture**: Features a dedicated Flask RESTful JSON API alongside an independent, modern React frontend.
2.  **MLflow Integration**: Tracks every model's hyperparameters and $R^2$ scores locally. Automatically logs the best-performing models to the MLflow registry.
3.  **Production Inference Logging**: All predictions made on the frontend are captured and permanently logged to a persistent SQLAlchemy SQLite database (`predictions.db`) for future drift analysis.
4.  **Containerized**: Fully deployable via `docker-compose`, spinning up the Backend, Frontend, and MLflow tracking server seamlessly.

## 🔧 Getting Started

### Option 1: Docker (Recommended)
You can spin up the entire microservices architecture with one command:
```bash
docker-compose up --build
```
This will expose:
- **React Frontend**: `http://localhost:5173`
- **Flask API**: `http://localhost:5000`
- **MLflow Tracking UI**: `http://localhost:5001`

### Option 2: Local Setup
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv_new
   .\venv_new\Scripts\activate
   pip install -r requirements.txt
   ```
2. Train the models and log to MLflow:
   ```bash
   python -m src.components.data_ingestion
   ```
3. Start the Flask Backend:
   ```bash
   python app.py
   ```
4. Start the React Frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## 📁 Repository Structure
```
├── artifacts/             # Serialized best models and preprocessors (.pkl)
├── frontend/              # Decoupled React/Vite UI
├── logs/                  # Application logs
├── mlruns/                # MLflow local tracking registry
├── src/                   # ML Pipeline source code (Ingestion, Transform, Training)
├── app.py                 # Flask REST API and SQLAlchemy Models
├── docker-compose.yml     # Multi-container orchestration
└── Dockerfile             # Backend container definition
```