"""
Database Models — SQLAlchemy ORM for prediction audit logging.

Using SQLite for local dev; easily swappable to PostgreSQL via
DATABASE_URL environment variable (production-ready pattern).
"""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

import os

# ---------------------------------------------------------------------------
# Database Setup
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./cognify.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class PredictionLog(Base):
    """
    Audit log for every inference event. Enables:
      - Analytics dashboard
      - Data drift detection
      - Retraining trigger monitoring
      - Model performance tracking over time
    """
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    request_id = Column(String(36), nullable=True)  # UUID for tracing

    # Input features
    gender = Column(String(10))
    race_ethnicity = Column(String(20))
    parental_level_of_education = Column(String(50))
    lunch = Column(String(20))
    test_preparation_course = Column(String(20))

    # Prediction outputs
    predicted_math_score = Column(Float)
    predicted_reading_score = Column(Float)
    predicted_writing_score = Column(Float)
    is_at_risk = Column(Boolean)
    career_recommendation = Column(String(50))
    is_anomaly = Column(Boolean)
    top_risk_factor = Column(String(100), nullable=True)

    # Metadata
    inference_latency_ms = Column(Float, nullable=True)
    model_version = Column(String(20), default="v2.4")


class RateLimitLog(Base):
    """Tracks API usage per key for rate limiting enforcement."""
    __tablename__ = "rate_limit_logs"

    id = Column(Integer, primary_key=True, index=True)
    api_key_hash = Column(String(64), index=True)
    endpoint = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45), nullable=True)


# ---------------------------------------------------------------------------
# DB Dependency (FastAPI pattern)
# ---------------------------------------------------------------------------

def get_db():
    """FastAPI dependency that yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called at application startup."""
    Base.metadata.create_all(bind=engine)
