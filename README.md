# COGNIFY — AI-Powered Student Performance & Academic Risk Intelligence Platform

## 1. Project Overview
Cognify is a production-grade, end-to-end Machine Learning decision-support platform designed for educational institutions. Rather than simply predicting a student's score, Cognify acts as an **early-warning system** that identifies students at risk of academic failure, explains the prediction using advanced ML interpretability (SHAP), and recommends personalized interventions.

## 2. The Real-World Problem
Educational institutions often identify struggling students only *after* their grades have already dropped—when it is too late for effective intervention. Traditional prediction systems only offer binary "pass/fail" or raw score predictions without explaining *why* the student is at risk or *what* can be done to help them.

## 3. The Solution
Cognify solves this by providing **Academic Risk Intelligence**:
- Estimates expected performance across multiple subjects (Math, Reading, Writing).
- Classifies the student's risk level (LOW, MEDIUM, HIGH).
- Explains the underlying causes for the risk using SHAP values.
- Simulates hypothetical scenarios (What-If Simulator).
- Prescribes targeted academic interventions based on detected behavioral weaknesses.

## 4. System Architecture
- **Frontend**: React, Tailwind CSS, Recharts (Dynamic Dashboard).
- **Backend**: FastAPI (Python), RESTful APIs.
- **ML Engine**: Scikit-Learn, XGBoost, Random Forest, SHAP.
- **Database**: SQLite (SQLAlchemy) for prediction history and analytics.
- **MLOps**: MLflow for experiment tracking and model registry.

## 5. ML Pipeline
1. **Data Ingestion**: Loads student demographic and behavioral data.
2. **Data Validation & Preprocessing**: Handles missing values (SimpleImputer) and scales numerical data (StandardScaler).
3. **Categorical Encoding**: One-Hot Encoding for demographic variables.
4. **Feature Engineering**: Synthetic behavioral features generation to enrich demographic data.
5. **Model Training**: Multi-Output Regression and Multi-Class Classification.
6. **Evaluation & Serialization**: Validation with R², saving model artifacts to the `artifacts/` directory.

## 6. Dataset
The platform uses an augmented dataset built on top of traditional student performance datasets.
- **Demographics**: Gender, Ethnicity, Parental Education, Lunch subsidy.
- **Test Preparation**: Completion status of preparatory courses.

## 7. Feature Engineering
Since demographic data alone is insufficient for a strong early-warning system, we engineered behavioral features:
- **Attendance Rate (%)**: Correlation with academic consistency.
- **Study Hours per Week**: Self-reported or logged study effort.
- **Previous GPA**: Historical performance baseline.
- **Assignment Completion Rate (%)**: Academic engagement indicator.

*Note: These behavioral features are synthetically generated for demonstration purposes. The system is designed to seamlessly integrate with real LMS (Learning Management System) data.*

## 8. Models
- **Performance Predictor**: XGBoost `MultiOutputRegressor` (Predicts Math, Reading, Writing, Overall Score).
- **Risk Classifier**: `RandomForestClassifier` (Multi-class: LOW, MEDIUM, HIGH) with balanced class weights to handle imbalance.
- **Career Clusterer**: `KMeans` clustering based on performance patterns.
- **Anomaly Detector**: `IsolationForest` to flag unusual student profiles.

## 9. Evaluation
- The Regression model is evaluated using **R²** and **RMSE**. 
- The Risk Classifier prioritizes **Recall** to minimize false negatives (failing to identify an at-risk student).
- Current Production R²: ~0.67 (A realistic metric for human behavioral data).

## 10. SHAP Explainability
Cognify implements **TreeExplainer** from the SHAP (SHapley Additive exPlanations) library to provide local interpretability. For every prediction, the platform identifies the `top_risk_factor` and its numerical impact, allowing educators to understand the *why* behind the AI's decision.

## 11. Risk Classification
Students are categorized into:
- **LOW RISK**: On track for academic success.
- **MEDIUM RISK**: Showing signs of academic struggle; monitor closely.
- **HIGH RISK**: High probability of underperformance; immediate intervention required.

## 12. What-If Simulation
The frontend includes a dynamic "What-If" simulator that allows counselors to adjust behavioral features (e.g., increasing study hours or attendance) and instantly see the model's estimated outcome. This helps visualize the potential impact of an intervention.

## 13. Personalized Intervention Engine
Rather than generic advice, the prescriptive engine generates recommendations based on the student's specific weaknesses. For example:
- If attendance < 80%: *"Improve attendance consistency."*
- If study hours < 10: *"Increase weekly study hours."*

## 14. API Documentation
The backend exposes documented REST APIs:
- `POST /api/v1/predict` - Run the full inference pipeline for a single student.
- `POST /api/v1/predict/batch` - Run batch predictions (up to 50 profiles).
- `GET /health` - API and model health status.

*(Swagger UI available at `http://localhost:8000/docs`)*

## 15. Database
- **SQLite Database** (`cognify.db`) stores a complete audit log of all predictions via SQLAlchemy.
- Stores inputs, predicted scores, risk levels, and SHAP factors for historical tracking and data drift analysis.

## 16. Frontend
A premium dark-themed React application featuring:
- Dynamic form inputs and sliders.
- Real-time Radar charts (Recharts).
- Clear visualization of SHAP factors and AI recommendations.
- Interactive AI Counselor chat interface.

## 17. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Cognify.git
cd Cognify

# Setup Backend
cd backend
python -m venv venv_new
venv_new\Scripts\activate  # Windows
source venv_new/bin/activate # Mac/Linux
pip install -r requirements.txt
cd ..

# Setup Frontend
cd frontend
npm install
cd ..
```

## 18. Environment Variables
Create a `.env` file in the `frontend` directory:
```env
VITE_API_URL=http://localhost:8000
```
Create a `.env` file in the `backend` directory (if deploying):
```env
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

## 19. Running Locally

**Terminal 1 (FastAPI Backend):**
```bash
cd backend
python main.py
# Server starts at http://localhost:8000
```

**Terminal 2 (React Frontend):**
```bash
cd frontend
npm run dev
# App starts at http://localhost:5173
```

## 20. Screenshots
*(Add screenshots of the Dashboard, What-If Simulator, and AI Counselor here)*

## 21. Limitations
- **Synthetic Data**: The behavioral features (`attendance`, `study_hours`) are synthetic, intended to demonstrate how the system would operate with real LMS data. Predictions should not be interpreted as causal.
- **Predictive, Not Deterministic**: The model provides statistical estimates, not guarantees. It is a decision-*support* tool, not a replacement for human academic judgment.

## 22. Future Improvements
- Integrate Deep Knowledge Tracing (DKT) for temporal sequence modeling.
- Implement live Data Drift monitoring using evidently.ai.
- Connect directly to Canvas or Moodle LMS APIs for real-time data ingestion.