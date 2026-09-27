import os
import sys
import pandas as pd
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

            preprocessor = load_object(file_path=preprocessor_path)
            radar_model = load_object(file_path=radar_model_path)
            atrisk_model = load_object(file_path=atrisk_model_path)
            cluster_model = load_object(file_path=cluster_model_path)

            # Preprocess features
            data_scaled = preprocessor.transform(features)
            
            # Convert to dense if sparse
            if hasattr(data_scaled, "toarray"):
                data_scaled = data_scaled.toarray()

            # 1. Multi-Output Regression (Radar)
            radar_preds = radar_model.predict(data_scaled)
            math, reading, writing = radar_preds[0]

            # 2. Binary Classification (At-Risk)
            atrisk_pred = atrisk_model.predict(data_scaled)[0]

            # 3. K-Means Clustering (Career) based on radar predictions
            # The cluster model expects shape (1, 3) representing [math, reading, writing]
            cluster_pred = cluster_model.predict([[math, reading, writing]])[0]
            
            # Mapping Cluster ID to Career Domain
            career_mapping = {
                0: "STEM (Engineering/Science)",
                1: "Business & Commerce",
                2: "Humanities & Arts"
            }
            recommended_career = career_mapping.get(int(cluster_pred), "General Academic")

            return {
                "math_score": float(math),
                "reading_score": float(reading),
                "writing_score": float(writing),
                "is_at_risk": bool(atrisk_pred),
                "career_recommendation": recommended_career
            }

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
