from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
from datetime import datetime
import os

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application
CORS(app)  # Enable CORS for the React frontend

# Configure SQLite Database
db_path = os.path.join(os.path.dirname(__file__), 'predictions.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define Database Model for Logging
class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    gender = db.Column(db.String(50))
    race_ethnicity = db.Column(db.String(50))
    parental_level_of_education = db.Column(db.String(100))
    lunch = db.Column(db.String(50))
    test_preparation_course = db.Column(db.String(50))
    reading_score = db.Column(db.Float)
    writing_score = db.Column(db.Float)
    predicted_math_score = db.Column(db.Float)

# Create the database tables
with app.app_context():
    db.create_all()

@app.route('/api/predict', methods=['POST'])
def predict_datapoint():
    try:
        # Support JSON for the React frontend
        req_data = request.get_json() if request.is_json else request.form

        gender = req_data.get('gender')
        race_ethnicity = req_data.get('race_ethnicity') or req_data.get('ethnicity')
        parental_level_of_education = req_data.get('parental_level_of_education')
        lunch = req_data.get('lunch')
        test_preparation_course = req_data.get('test_preparation_course')
        reading_score = float(req_data.get('reading_score'))
        writing_score = float(req_data.get('writing_score'))

        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        pred_df = data.get_data_as_data_frame()
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        predicted_score = float(results[0])
        print("After Prediction")

        # Log to the Database
        new_log = PredictionLog(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score,
            predicted_math_score=predicted_score
        )
        db.session.add(new_log)
        db.session.commit()
        
        return jsonify({'prediction': predicted_score})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
