"""
Pydantic Schemas — Request & Response Models for the Cognify API.

All data flowing in and out of the API is strictly validated and
documented here. This enables auto-generated Swagger docs and
ensures type-safety throughout the entire ML inference pipeline.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import enum


# ---------------------------------------------------------------------------
# Enums for categorical inputs
# ---------------------------------------------------------------------------

class GenderEnum(str, enum.Enum):
    male = "male"
    female = "female"


class EthnicityEnum(str, enum.Enum):
    group_a = "group A"
    group_b = "group B"
    group_c = "group C"
    group_d = "group D"
    group_e = "group E"


class EducationEnum(str, enum.Enum):
    some_high_school = "some high school"
    high_school = "high school"
    some_college = "some college"
    associates = "associate's degree"
    bachelors = "bachelor's degree"
    masters = "master's degree"


class LunchEnum(str, enum.Enum):
    standard = "standard"
    free_reduced = "free/reduced"


class TestPrepEnum(str, enum.Enum):
    none = "none"
    completed = "completed"


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class StudentProfileRequest(BaseModel):
    gender: GenderEnum = Field(..., description="Student's gender", examples=["male"])
    race_ethnicity: EthnicityEnum = Field(..., description="Student's ethnic group", examples=["group C"])
    parental_level_of_education: EducationEnum = Field(..., description="Highest level of education", examples=["bachelor's degree"])
    lunch: LunchEnum = Field(..., description="Lunch subsidy", examples=["standard"])
    test_preparation_course: TestPrepEnum = Field(..., description="Test prep", examples=["none"])
    
    # Behavioral Features
    attendance_rate: float = Field(default=85.0, description="Attendance percentage (0-100)", examples=[85.0])
    study_hours_per_week: float = Field(default=10.0, description="Study hours per week (0-40)", examples=[10.0])
    previous_gpa: float = Field(default=3.2, description="Previous GPA (1.0-4.0)", examples=[3.2])
    assignment_completion_rate: float = Field(default=90.0, description="Assignment completion rate (0-100)", examples=[90.0])

    # Backward compatible fields (can be deprecated)
    highest_education: str = Field(default="A Level or Equivalent", description="Student's highest education")
    num_of_prev_attempts: int = Field(default=0, description="Previous attempts")
    studied_credits: int = Field(default=60, description="Studied credits")
    total_clicks: float = Field(default=15.0, description="Total clicks (Weeks 1-4)")
    avg_assessment_score: float = Field(default=55.0, description="Average assessment score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "female",
                "race_ethnicity": "group C",
                "parental_level_of_education": "bachelor's degree",
                "lunch": "standard",
                "test_preparation_course": "completed",
                "attendance_rate": 85.0,
                "study_hours_per_week": 10.0,
                "previous_gpa": 3.2,
                "assignment_completion_rate": 90.0,
                "highest_education": "A Level or Equivalent",
                "num_of_prev_attempts": 0,
                "studied_credits": 60,
                "total_clicks": 15.0,
                "avg_assessment_score": 55.0
            }
        }
    }



class BatchPredictionRequest(BaseModel):
    """Batch inference — up to 50 student profiles in a single request."""
    students: list[StudentProfileRequest] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of student profiles (max 50)"
    )


class ChatRequest(BaseModel):
    """Request schema for the AI Counselor chat endpoint."""
    message: str = Field(..., min_length=1, max_length=500, description="User's message")
    context: Optional[dict] = Field(
        default=None,
        description="ML prediction context to ground the AI response"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What career path is best for me?",
                "context": {
                    "math_score": 72.5,
                    "reading_score": 68.1,
                    "writing_score": 65.4,
                    "is_at_risk": False,
                    "career_recommendation": "STEM (Engineering/Science)"
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# Response Schemas
# ---------------------------------------------------------------------------

class DKTTrajectoryResponse(BaseModel):
    """Deep Knowledge Tracing — simulated learning trajectory across 3 time steps."""
    math_knowledge: list[float]
    reading_knowledge: list[float]
    writing_knowledge: list[float]


class PredictionResponse(BaseModel):
    """
    Full inference response from the 5-engine ML pipeline.
    Every field is documented for auto-generated API docs.
    """
    # Core regression scores
    math_score: float = Field(..., description="Predicted math score (0-100)")
    reading_score: float = Field(..., description="Predicted reading score (0-100)")
    writing_score: float = Field(..., description="Predicted writing score (0-100)")
    overall_score: float = Field(..., description="Predicted overall academic score (0-100)")

    # Classification
    is_at_risk: bool = Field(..., description="True if student is at risk of academic failure")
    risk_level: str = Field(..., description="Academic risk level: LOW, MEDIUM, or HIGH")

    # Clustering
    career_recommendation: str = Field(..., description="K-Means career cluster recommendation")

    # Anomaly Detection
    is_anomaly: bool = Field(..., description="True if Isolation Forest flagged an unusual profile")

    # Explainability
    top_risk_factor: str = Field(..., description="SHAP-identified primary risk feature")
    top_risk_impact: float = Field(..., description="SHAP impact value of the top risk feature")

    # Deep Knowledge Tracing
    dkt_trajectory: DKTTrajectoryResponse

    # Prescriptive AI
    advisor_message: str = Field(..., description="What-If engine prescriptive recommendation")

    # Metadata
    prediction_id: int = Field(..., description="Database log ID for this inference event")
    inference_latency_ms: float = Field(..., description="End-to-end ML inference time in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Response for a batch inference request."""
    total: int
    predictions: list[PredictionResponse]
    batch_latency_ms: float


class AnalyticsLogEntry(BaseModel):
    """A single row from the predictions audit log."""
    id: int
    timestamp: str
    gender: str
    predicted_math_score: float
    predicted_reading_score: float
    predicted_writing_score: float
    is_at_risk: bool
    career_recommendation: str


class AnalyticsResponse(BaseModel):
    """Summary analytics from the prediction audit log."""
    total_predictions: int
    at_risk_count: int
    at_risk_rate: float
    avg_math_score: float
    avg_reading_score: float
    avg_writing_score: float
    drift_warning: bool
    drift_message: str
    recent_logs: list[AnalyticsLogEntry]


class ModelMetaResponse(BaseModel):
    """Metadata about all active ML models."""
    radar_model: str
    atrisk_model: str
    cluster_model: str
    anomaly_model: str
    shap_explainer: str
    training_pipeline: str
    framework: str
    version: str


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    models_loaded: bool
    database_ok: bool
    version: str
    timestamp: str


class ChatResponse(BaseModel):
    """Response from the AI Counselor endpoint."""
    reply: str
    model_used: str


class ErrorResponse(BaseModel):
    """Standard error response envelope."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
