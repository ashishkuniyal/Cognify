import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    radar_model_file_path=os.path.join("artifacts","model_radar.pkl")
    atrisk_model_file_path=os.path.join("artifacts","model_atrisk.pkl")
    cluster_model_file_path=os.path.join("artifacts","model_cluster.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data for V2")
            
            # X features are all columns except the last 4 (math, reading, writing, at_risk)
            X_train = train_array[:, :-4]
            X_test = test_array[:, :-4]
            
            # Target 1: Radar (Math, Reading, Writing)
            y_radar_train = train_array[:, -4:-1]
            y_radar_test = test_array[:, -4:-1]
            
            # Target 2: At-Risk (Binary)
            y_atrisk_train = train_array[:, -1]
            y_atrisk_test = test_array[:, -1]

            import mlflow
            import mlflow.sklearn

            mlflow.set_experiment("Student_Performance_V2")
            
            with mlflow.start_run(run_name="V2_Multi_Engine_Training"):
                
                # 1. Train Radar Model
                logging.info("Training Radar MultiOutput Regressor")
                radar_model = MultiOutputRegressor(XGBRegressor(learning_rate=0.1, n_estimators=100))
                radar_model.fit(X_train, y_radar_train)
                
                radar_preds = radar_model.predict(X_test)
                radar_r2 = r2_score(y_radar_test, radar_preds)
                mlflow.log_metric("radar_r2_score", radar_r2)
                
                # 2. Train At-Risk Classifier
                logging.info("Training At-Risk Classifier")
                atrisk_model = RandomForestClassifier(n_estimators=100, random_state=42)
                atrisk_model.fit(X_train, y_atrisk_train)
                
                atrisk_preds = atrisk_model.predict(X_test)
                atrisk_acc = accuracy_score(y_atrisk_test, atrisk_preds)
                mlflow.log_metric("atrisk_accuracy", atrisk_acc)
                
                # 3. Train Career Clusterer
                # Note: We cluster based on the student's scores (y_radar), not inputs
                logging.info("Training Career Clusterer")
                cluster_model = KMeans(n_clusters=3, random_state=42)
                cluster_model.fit(y_radar_train)

                # Save all models
                logging.info("Saving V2 Models")
                save_object(file_path=self.model_trainer_config.radar_model_file_path, obj=radar_model)
                save_object(file_path=self.model_trainer_config.atrisk_model_file_path, obj=atrisk_model)
                save_object(file_path=self.model_trainer_config.cluster_model_file_path, obj=cluster_model)
                
                return radar_r2

        except Exception as e:
            raise CustomException(e,sys)