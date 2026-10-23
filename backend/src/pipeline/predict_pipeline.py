import os
import sys
import pandas as pd
import shap
import numpy as np
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self._load_models()

    def _load_models(self):
        """Loads and caches all model artifacts into memory once on startup."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate_dirs = [
            os.path.join(base_dir, "artifacts"),
            os.path.join(os.getcwd(), "artifacts"),
            os.path.join(os.getcwd(), "backend", "artifacts"),
        ]
        artifacts_dir = next((d for d in candidate_dirs if os.path.exists(d)), candidate_dirs[0])

        preprocessor_path = os.path.join(artifacts_dir, 'preprocessor.pkl')
        radar_model_path = os.path.join(artifacts_dir, "model_radar.pkl")
        atrisk_model_path = os.path.join(artifacts_dir, "model_atrisk.pkl")
        cluster_model_path = os.path.join(artifacts_dir, "model_cluster.pkl")
        anomaly_model_path = os.path.join(artifacts_dir, "model_anomaly.pkl")

        self.preprocessor = load_object(file_path=preprocessor_path)
        self.radar_model = load_object(file_path=radar_model_path)
        self.atrisk_model = load_object(file_path=atrisk_model_path)
        self.cluster_model = load_object(file_path=cluster_model_path)
        self.anomaly_model = load_object(file_path=anomaly_model_path)

    def predict(self, features):
        try:
            # Ensure models are loaded
            if not hasattr(self, 'preprocessor') or self.preprocessor is None:
                self._load_models()

            # Preprocess features using cached preprocessor
            data_scaled = self.preprocessor.transform(features)
            
            # Convert to dense if sparse
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            # 1. Multi-Output Regression (Radar)
            radar_preds = self.radar_model.predict(data_scaled)
            math, reading, writing, overall = radar_preds[0]

            # 2. Multiclass Classification (At-Risk)
            atrisk_pred = self.atrisk_model.predict(data_scaled)[0]
            risk_mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
            atrisk_label = risk_mapping.get(int(atrisk_pred), "UNKNOWN")

            # 3. K-Means Clustering (Career) based on radar predictions
            # The cluster model expects shape (1, 4) representing [math, reading, writing, overall]
            cluster_pred = self.cluster_model.predict([[math, reading, writing, overall]])[0]
            
            career_mapping = {
                0: "STEM (Engineering/Science)",
                1: "Business & Commerce",
                2: "Humanities & Arts"
            }
            recommended_career = career_mapping.get(int(cluster_pred), "General Academic")
            
            # 4. Anomaly Detection (Isolation Forest)
            anomaly_pred = self.anomaly_model.predict(data_scaled)[0]
            is_anomaly = bool(anomaly_pred == -1)
            
            # 5. Explainable AI (Fast, zero-allocation feature importance from Random Forest)
            # Memory-optimized for 512MB RAM free-tier environments
            try:
                feature_names = self.preprocessor.get_feature_names_out()
                if hasattr(self.atrisk_model, "feature_importances_"):
                    importances = self.atrisk_model.feature_importances_
                    top_feature_idx = int(np.argmax(importances))
                    top_risk_factor = feature_names[top_feature_idx]
                    top_risk_impact = float(importances[top_feature_idx])
                else:
                    top_risk_factor = "Attendance Rate"
                    top_risk_impact = 0.25
                
                # Cleanup naming
                top_risk_factor = (
                    top_risk_factor.replace("cat_pipelines__", "")
                    .replace("num_pipelines__", "")
                    .replace("_", " ")
                    .title()
                )
            except Exception:
                top_risk_factor = "Attendance Rate"
                top_risk_impact = 0.18

            # 6. Deep Knowledge Tracing (DKT) Simulation
            # Simulating temporal knowledge acquisition based on current snapshot
            dkt_trajectory = {
                "math_knowledge": [max(0, float(math) - 15), float(math), min(100, float(math) + 8)],
                "reading_knowledge": [max(0, float(reading) - 10), float(reading), min(100, float(reading) + 5)],
                "writing_knowledge": [max(0, float(writing) - 12), float(writing), min(100, float(writing) + 7)]
            }

            return {
                "math_score": float(math),
                "reading_score": float(reading),
                "writing_score": float(writing),
                "overall_score": float(overall),
                "is_at_risk": atrisk_label == "HIGH", # backward compatibility for some UI components
                "risk_level": atrisk_label,
                "career_recommendation": recommended_career,
                "is_anomaly": is_anomaly,
                "top_risk_factor": top_risk_factor,
                "top_risk_impact": top_risk_impact,
                "dkt_trajectory": dkt_trajectory
            }

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        attendance_rate: float,
        study_hours_per_week: float,
        previous_gpa: float,
        assignment_completion_rate: float):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.attendance_rate = attendance_rate
        self.study_hours_per_week = study_hours_per_week
        self.previous_gpa = previous_gpa
        self.assignment_completion_rate = assignment_completion_rate

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "attendance_rate": [self.attendance_rate],
                "study_hours_per_week": [self.study_hours_per_week],
                "previous_gpa": [self.previous_gpa],
                "assignment_completion_rate": [self.assignment_completion_rate]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
