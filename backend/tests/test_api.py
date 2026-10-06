import os
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "degraded"]

def test_predict_endpoint_validation():
    # Missing fields should fail validation
    payload = {
        "gender": "male"
    }
    response = client.post("/api/v1/predict", json=payload, headers={"X-API-Key": "dev-key-2026"})
    assert response.status_code == 422 # Pydantic validation error

def test_predict_endpoint_success():
    payload = {
        "gender": "female",
        "race_ethnicity": "group C",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "none",
        "attendance_rate": 85.0,
        "study_hours_per_week": 12.0,
        "previous_gpa": 3.2,
        "assignment_completion_rate": 90.0
    }
    response = client.post("/api/v1/predict", json=payload, headers={"X-API-Key": "dev-key-2026"})
    
    # If ML models are loaded, it should return 200, else 503
    if response.status_code == 200:
        data = response.json()
        assert "math_score" in data
        assert "overall_score" in data
        assert "risk_level" in data
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert "advisor_message" in data
    else:
        assert response.status_code == 503
