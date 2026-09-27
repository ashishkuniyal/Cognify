from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_session import Session
import numpy as np
import pandas as pd
from datetime import datetime
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv(override=True)

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application
CORS(app, supports_credentials=True)  # Enable CORS with credentials for sessions

# Configure Flask Session (SQLAlchemy-backed, works on cloud platforms)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'student-matrix-v2-secret-key-2024')
app.config['SESSION_TYPE'] = 'sqlalchemy'
app.config['SESSION_PERMANENT'] = False

# Configure SQLite Database
db_path = os.path.join(os.path.dirname(__file__), 'predictions.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.config['SESSION_SQLALCHEMY'] = db
Session(app)

# Define Database Model for Logging
class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    gender = db.Column(db.String(50))
    race_ethnicity = db.Column(db.String(50))
    parental_level_of_education = db.Column(db.String(100))
    lunch = db.Column(db.String(50))
    test_preparation_course = db.Column(db.String(50))
    predicted_math_score = db.Column(db.Float)
    predicted_reading_score = db.Column(db.Float)
    predicted_writing_score = db.Column(db.Float)
    is_at_risk = db.Column(db.Boolean)
    career_recommendation = db.Column(db.String(100))

# Create the database tables
with app.app_context():
    db.create_all()

@app.route('/api/predict', methods=['POST'])
def predict_datapoint():
    try:
        req_data = request.get_json() if request.is_json else request.form

        gender = req_data.get('gender')
        race_ethnicity = req_data.get('race_ethnicity') or req_data.get('ethnicity')
        parental_level_of_education = req_data.get('parental_level_of_education')
        lunch = req_data.get('lunch')
        test_preparation_course = req_data.get('test_preparation_course')

        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course
        )
        pred_df = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        
        # Prescriptive AI (What-If Engine)
        advisor_message = "You are currently maximizing your potential."
        if test_preparation_course != "completed":
            # Run simulation
            sim_df = pred_df.copy()
            sim_df['test_preparation_course'] = 'completed'
            sim_results = predict_pipeline.predict(sim_df)
            
            math_boost = sim_results['math_score'] - results['math_score']
            if math_boost > 0:
                advisor_message = f"Completing a test preparation course could boost your Math score by +{math_boost:.1f} points!"

        results["advisor_message"] = advisor_message

        # Log to the Database
        new_log = PredictionLog(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            predicted_math_score=results['math_score'],
            predicted_reading_score=results['reading_score'],
            predicted_writing_score=results['writing_score'],
            is_at_risk=results['is_at_risk'],
            career_recommendation=results['career_recommendation']
        )
        db.session.add(new_log)
        db.session.commit()
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    try:
        logs = PredictionLog.query.order_by(PredictionLog.timestamp.desc()).limit(50).all()
        data = []
        for log in logs:
            data.append({
                'id': log.id,
                'timestamp': log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'predicted_math_score': round(log.predicted_math_score, 2),
                'is_at_risk': log.is_at_risk,
                'career_recommendation': log.career_recommendation
            })
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        req_data = request.get_json()
        message = req_data.get('message', '')
        context = req_data.get('context') or {}
        reset = req_data.get('reset', False)
        
        if not context:
            return jsonify({"reply": "I'd love to help! Please go to the **Prediction** tab first and click **Run Prediction Model** so I can analyze your academic profile."})
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            return jsonify({"reply": "Please add a valid GEMINI_API_KEY to the .env file and restart the backend."})
        
        # Build the system instruction with ML context
        system_instruction = f"""You are an expert AI Academic and Career Counselor embedded inside the Student Matrix V2 EdTech platform.

The student's profile has been analyzed by our Machine Learning engines and here are their results:
- Math Score (predicted): {round(context.get('math_score', 0))}/100
- Reading Score (predicted): {round(context.get('reading_score', 0))}/100
- Writing Score (predicted): {round(context.get('writing_score', 0))}/100
- At Risk of Academic Failure: {'YES - needs urgent support' if context.get('is_at_risk') else 'No - performing adequately'}
- ML-Recommended Career Cluster: {context.get('career_recommendation', 'General Academic')}
- Prescriptive Insight: {context.get('advisor_message', 'Keep up the consistent effort.')}

Your role:
- Give warm, encouraging, and personalized advice based strictly on the profile above.
- Keep answers concise (2-4 sentences max).
- Use their actual scores and career path when giving recommendations.
- If they are at risk, be empathetic but provide a clear, actionable step."""

        # Retrieve or initialize conversation history from server-side session
        if reset or 'chat_history' not in session:
            session['chat_history'] = []
        
        history = session['chat_history']
        
        client = genai.Client(api_key=api_key)
        
        # Reconstruct the Gemini chat with full history for true multi-turn memory
        gemini_history = []
        for turn in history:
            gemini_history.append(types.Content(role=turn['role'], parts=[types.Part(text=turn['text'])]))
        
        # Model fallback cascade — try newer models first, fall back gracefully
        model_cascade = ['gemini-3.8-flash', 'gemini-3.5-flash', 'gemini-flash-latest']
        reply_text = None
        last_error = None
        
        for model_name in model_cascade:
            try:
                chat_session = client.chats.create(
                    model=model_name,
                    config=types.GenerateContentConfig(system_instruction=system_instruction),
                    history=gemini_history
                )
                response = chat_session.send_message(message)
                reply_text = response.text
                break  # Success — stop trying
            except Exception as model_err:
                last_error = model_err
                err_str = str(model_err)
                # Only continue cascade on availability/quota errors
                if '503' in err_str or '429' in err_str or 'UNAVAILABLE' in err_str or 'not found' in err_str.lower() or 'NOT_FOUND' in err_str:
                    continue
                raise  # Re-raise unexpected errors immediately
        
        if reply_text is None:
            return jsonify({"reply": f"All AI models are currently experiencing high demand. Please try again in a moment. (Last error: {str(last_error)[:100]})"}), 503
        
        # Persist updated history to session
        history.append({'role': 'user', 'text': message})
        history.append({'role': 'model', 'text': reply_text})
        session['chat_history'] = history
        session.modified = True
        
        return jsonify({"reply": reply_text})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/chat/reset', methods=['POST'])
def reset_chat():
    session.pop('chat_history', None)
    return jsonify({'status': 'ok'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
