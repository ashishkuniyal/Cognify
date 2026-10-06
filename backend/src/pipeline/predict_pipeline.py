import os
import sys
import pandas as pd
import shap
import numpy as np
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            radar_model_path = os.path.join("artifacts", "model_radar.pkl")
            atrisk_model_path = os.path.join("artifacts", "model_atrisk.pkl")
            cluster_model_path = os.path.join("artifacts", "model_cluster.pkl")
            anomaly_model_path = os.path.join("artifacts", "model_anomaly.pkl")

            preprocessor = load_object(file_path=preprocessor_path)
            radar_model = load_object(file_path=radar_model_path)
            atrisk_model = load_object(file_path=atrisk_model_path)
            cluster_model = load_object(file_path=cluster_model_path)
            anomaly_model = load_object(file_path=anomaly_model_path)

            # Preprocess features
            data_scaled = preprocessor.transform(features)
            
            # Convert to dense if sparse
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            # 1. Multi-Output Regression (Radar)
            radar_preds = radar_model.predict(data_scaled)
            math, reading, writing, overall = radar_preds[0]

            # 2. Multiclass Classification (At-Risk)
            atrisk_pred = atrisk_model.predict(data_scaled)[0]
            risk_mapping = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
            atrisk_label = risk_mapping.get(int(atrisk_pred), "UNKNOWN")

            # 3. K-Means Clustering (Career) based on radar predictions
            # The cluster model expects shape (1, 4) representing [math, reading, writing, overall]
            cluster_pred = cluster_model.predict([[math, reading, writing, overall]])[0]
            
            career_mapping = {
                0: "STEM (Engineering/Science)",
                1: "Business & Commerce",
                2: "Humanities & Arts"
            }
            recommended_career = career_mapping.get(int(cluster_pred), "General Academic")
            
            # 4. Anomaly Detection (Isolation Forest)
            anomaly_pred = anomaly_model.predict(data_scaled)[0]
            is_anomaly = bool(anomaly_pred == -1)
            
            # 5. Explainable AI (SHAP)
            try:
                explainer = shap.TreeExplainer(atrisk_model)
                shap_values = explainer.shap_values(data_scaled)
                pred_class_idx = int(atrisk_pred)
                
                if isinstance(shap_values, list):
                    # For multi-class RF, shap_values is a list of arrays (one for each class)
                    shap_vals = shap_values[pred_class_idx][0]
                elif len(shap_values.shape) == 3:
                    # Depending on shap version, it might return a 3D array (samples, features, classes)
                    shap_vals = shap_values[0, :, pred_class_idx]
                else:
                    shap_vals = shap_values[0]
                    
                feature_names = preprocessor.get_feature_names_out()
                top_feature_idx = np.abs(shap_vals).argmax()
                top_risk_factor = feature_names[top_feature_idx]
                top_risk_impact = float(shap_vals[top_feature_idx])
                
                # Cleanup naming
                top_risk_factor = top_risk_factor.replace("cat_pipelines__", "").replace("num_pipelines__", "").replace("_", " ").title()
            except Exception as e:
                top_risk_factor = "Unknown"
                top_risk_impact = 0.0

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
