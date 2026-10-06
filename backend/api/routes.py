"""
API Routers — All endpoint definitions organized by domain.

Each router is a self-contained FastAPI APIRouter mounted at /api/v1/.
This mirrors the production pattern used at AI-driven companies.

Routers:
  /api/v1/predict      — Single & batch ML inference
  /api/v1/analytics    — Prediction audit log & drift detection
  /api/v1/chat         — AI Counselor (Gemini-powered)
  /api/v1/models       — Model metadata & health
"""
from __future__ import annotations

import json
import logging
import os
import statistics
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from api.database import PredictionLog, get_db
from api.middleware import apply_rate_limit, get_api_key
from api.ml_service import get_ml_service, MLService
from api.schemas import (
    AnalyticsLogEntry,
    AnalyticsResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChatRequest,
    ChatResponse,
    DKTTrajectoryResponse,
    ErrorResponse,
    ModelMetaResponse,
    PredictionResponse,
    StudentProfileRequest,
)

logger = logging.getLogger("cognify.routes")


# =============================================================================
# PREDICTION ROUTER
# =============================================================================

prediction_router = APIRouter(prefix="/predict", tags=["Prediction Engine"])


@prediction_router.post(
    "",
    response_model=PredictionResponse,
    summary="Single Student Prediction",
    description=(
        "Run the full **5-engine ML pipeline** on a single student profile. "
        "Returns predicted scores, at-risk classification, career cluster, "
        "anomaly flag, SHAP risk explanation, DKT trajectory, and a prescriptive AI message."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Validation or inference error"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        503: {"model": ErrorResponse, "description": "Models not loaded yet"},
    }
)
async def predict_single(
    payload: StudentProfileRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
    ml: MLService = Depends(get_ml_service),
    db: Session = Depends(get_db),
):
    apply_rate_limit(request, api_key, "predict")

    if not ml.models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML models are not yet loaded. Please check server logs."
        )

    try:
        results = ml.predict_single(payload.model_dump())
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Persist to audit log
    request_id = getattr(request.state, "request_id", None)
    log_entry = PredictionLog(
        request_id=request_id,
        gender=payload.gender.value,
        race_ethnicity=payload.race_ethnicity.value,
        parental_level_of_education=payload.parental_level_of_education.value,
        lunch=payload.lunch.value,
        test_preparation_course=payload.test_preparation_course.value,
        predicted_math_score=results["math_score"],
        predicted_reading_score=results["reading_score"],
        predicted_writing_score=results["writing_score"],
        is_at_risk=results["is_at_risk"],
        career_recommendation=results["career_recommendation"],
        is_anomaly=results["is_anomaly"],
        top_risk_factor=results.get("top_risk_factor"),
        inference_latency_ms=results["inference_latency_ms"],
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    return PredictionResponse(
        math_score=results["math_score"],
        reading_score=results["reading_score"],
        writing_score=results["writing_score"],
        overall_score=results["overall_score"],
        is_at_risk=results["is_at_risk"],
        risk_level=results["risk_level"],
        career_recommendation=results["career_recommendation"],
        is_anomaly=results["is_anomaly"],
        top_risk_factor=results["top_risk_factor"],
        top_risk_impact=results["top_risk_impact"],
        dkt_trajectory=DKTTrajectoryResponse(**results["dkt_trajectory"]),
        advisor_message=results["advisor_message"],
        prediction_id=log_entry.id,
        inference_latency_ms=results["inference_latency_ms"],
    )


@prediction_router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Batch Student Prediction (up to 50)",
    description=(
        "Submit up to **50 student profiles** in a single API call. "
        "Each profile is processed through the full ML pipeline independently. "
        "Ideal for bulk analysis of classroom or cohort data."
    ),
)
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
    ml: MLService = Depends(get_ml_service),
    db: Session = Depends(get_db),
):
    import time
    apply_rate_limit(request, api_key, "batch")

    if not ml.models_loaded:
        raise HTTPException(status_code=503, detail="ML models not loaded.")

    start = time.perf_counter()
    try:
        all_results = ml.predict_batch([p.model_dump() for p in payload.students])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    batch_latency = (time.perf_counter() - start) * 1000
    responses = []

    for i, (student, results) in enumerate(zip(payload.students, all_results)):
        request_id = getattr(request.state, "request_id", None)
        log_entry = PredictionLog(
            request_id=f"{request_id}-{i}" if request_id else None,
            gender=student.gender.value,
            race_ethnicity=student.race_ethnicity.value,
            parental_level_of_education=student.parental_level_of_education.value,
            lunch=student.lunch.value,
            test_preparation_course=student.test_preparation_course.value,
            predicted_math_score=results["math_score"],
            predicted_reading_score=results["reading_score"],
            predicted_writing_score=results["writing_score"],
            is_at_risk=results["is_at_risk"],
            career_recommendation=results["career_recommendation"],
            is_anomaly=results["is_anomaly"],
            top_risk_factor=results.get("top_risk_factor"),
            inference_latency_ms=results["inference_latency_ms"],
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

        responses.append(PredictionResponse(
            math_score=results["math_score"],
            reading_score=results["reading_score"],
            writing_score=results["writing_score"],
            overall_score=results["overall_score"],
            is_at_risk=results["is_at_risk"],
            risk_level=results["risk_level"],
            career_recommendation=results["career_recommendation"],
            is_anomaly=results["is_anomaly"],
            top_risk_factor=results["top_risk_factor"],
            top_risk_impact=results["top_risk_impact"],
            dkt_trajectory=DKTTrajectoryResponse(**results["dkt_trajectory"]),
            advisor_message=results["advisor_message"],
            prediction_id=log_entry.id,
            inference_latency_ms=results["inference_latency_ms"],
        ))

    return BatchPredictionResponse(
        total=len(responses),
        predictions=responses,
        batch_latency_ms=round(batch_latency, 2),
    )


# =============================================================================
# ANALYTICS ROUTER
# =============================================================================

analytics_router = APIRouter(prefix="/analytics", tags=["Analytics & Monitoring"])


@analytics_router.get(
    "",
    response_model=AnalyticsResponse,
    summary="Prediction Analytics Dashboard",
    description=(
        "Retrieve aggregated analytics from the prediction audit log. "
        "Includes at-risk rates, average scores, and **data drift detection** "
        "using a simple statistical threshold (avg score < 45 or > 85 triggers a warning)."
    ),
)
async def get_analytics(
    limit: int = 50,
    api_key: str = Depends(get_api_key),
    db: Session = Depends(get_db),
):
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )

    if not logs:
        return AnalyticsResponse(
            total_predictions=0,
            at_risk_count=0,
            at_risk_rate=0.0,
            avg_math_score=0.0,
            avg_reading_score=0.0,
            avg_writing_score=0.0,
            drift_warning=False,
            drift_message="No prediction data yet.",
            recent_logs=[],
        )

    math_scores = [l.predicted_math_score for l in logs if l.predicted_math_score is not None]
    reading_scores = [l.predicted_reading_score for l in logs if l.predicted_reading_score is not None]
    writing_scores = [l.predicted_writing_score for l in logs if l.predicted_writing_score is not None]
    at_risk_count = sum(1 for l in logs if l.is_at_risk)

    avg_math = statistics.mean(math_scores) if math_scores else 0.0
    avg_reading = statistics.mean(reading_scores) if reading_scores else 0.0
    avg_writing = statistics.mean(writing_scores) if writing_scores else 0.0

    drift_warning = avg_math < 45 or avg_math > 85
    drift_message = (
        f"⚠️ Data Drift Detected: avg Math score ({avg_math:.1f}) is outside expected range [45–85]."
        if drift_warning
        else "✅ No data drift detected. Model is performing within expected bounds."
    )

    recent_logs = [
        AnalyticsLogEntry(
            id=l.id,
            timestamp=l.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            gender=l.gender or "N/A",
            predicted_math_score=round(l.predicted_math_score or 0, 2),
            predicted_reading_score=round(l.predicted_reading_score or 0, 2),
            predicted_writing_score=round(l.predicted_writing_score or 0, 2),
            is_at_risk=l.is_at_risk or False,
            career_recommendation=l.career_recommendation or "N/A",
        )
        for l in logs
    ]

    return AnalyticsResponse(
        total_predictions=len(logs),
        at_risk_count=at_risk_count,
        at_risk_rate=round(at_risk_count / len(logs), 4),
        avg_math_score=round(avg_math, 2),
        avg_reading_score=round(avg_reading, 2),
        avg_writing_score=round(avg_writing, 2),
        drift_warning=drift_warning,
        drift_message=drift_message,
        recent_logs=recent_logs,
    )


# =============================================================================
# CHAT ROUTER
# =============================================================================

chat_router = APIRouter(prefix="/chat", tags=["AI Counselor"])

# In-memory chat history store (keyed by API key hash)
_chat_sessions: dict[str, list] = {}


@chat_router.post(
    "",
    response_model=ChatResponse,
    summary="AI Academic Counselor (Gemini-powered)",
    description=(
        "Send a message to the **Gemini-powered AI Counselor**. "
        "Pass the `context` field with the student's ML prediction output "
        "to enable personalized, profile-aware responses. "
        "Chat history is maintained per API key for multi-turn conversations."
    ),
)
async def chat(
    payload: ChatRequest,
    request: Request,
    api_key: str = Depends(get_api_key),
):
    apply_rate_limit(request, api_key, "chat")

    context = payload.context or {}
    if not context:
        return ChatResponse(
            reply="Please run a prediction first so I can analyze your academic profile.",
            model_used="none"
        )

    api_key_env = os.environ.get("GEMINI_API_KEY", "")
    if not api_key_env or api_key_env == "YOUR_API_KEY_HERE":
        return ChatResponse(
            reply="Please add a valid GEMINI_API_KEY to the .env file.",
            model_used="none"
        )

    from google import genai
    from google.genai import types

    system_instruction = f"""You are an expert AI Academic and Career Counselor inside the Cognify EdTech platform.

The student's ML profile:
- Math Score: {round(context.get('math_score', 0))}/100
- Reading Score: {round(context.get('reading_score', 0))}/100
- Writing Score: {round(context.get('writing_score', 0))}/100
- At Risk: {'YES — needs urgent support' if context.get('is_at_risk') else 'No'}
- Primary Risk Factor (SHAP): {context.get('top_risk_factor', 'N/A')}
- Anomalous Profile: {'YES' if context.get('is_anomaly') else 'No'}
- Career Cluster: {context.get('career_recommendation', 'General Academic')}
- Prescriptive Insight: {context.get('advisor_message', 'Keep up consistent effort.')}

Rules: Be warm, concise (2-4 sentences), empathetic, and action-oriented."""

    # Retrieve session history per API key
    import hashlib
    session_key = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    history = _chat_sessions.get(session_key, [])

    client = genai.Client(api_key=api_key_env)
    model_cascade = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    reply_text = None
    model_used = "unknown"
    last_error = None

    gemini_history = [
        types.Content(role=t["role"], parts=[types.Part(text=t["text"])])
        for t in history
    ]

    for model_name in model_cascade:
        try:
            chat_session = client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(system_instruction=system_instruction),
                history=gemini_history,
            )
            response = chat_session.send_message(payload.message)
            reply_text = response.text
            model_used = model_name
            break
        except Exception as e:
            last_error = e
            err = str(e)
            if any(code in err for code in ["503", "429", "UNAVAILABLE", "not found", "NOT_FOUND"]):
                continue
            raise HTTPException(status_code=500, detail=str(e))

    if reply_text is None:
        return ChatResponse(
            reply=f"AI models are experiencing high demand. Last error: {str(last_error)[:80]}",
            model_used="error"
        )

    # Update session history
    history.append({"role": "user", "text": payload.message})
    history.append({"role": "model", "text": reply_text})
    _chat_sessions[session_key] = history[-20:]  # keep last 10 turns

    return ChatResponse(reply=reply_text, model_used=model_used)


@chat_router.delete(
    "/session",
    summary="Clear Chat Memory",
    description="Wipes the multi-turn chat history for the current API key session.",
)
async def clear_chat_session(api_key: str = Depends(get_api_key)):
    import hashlib
    session_key = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    _chat_sessions.pop(session_key, None)
    return {"status": "ok", "message": "Chat memory cleared."}


# =============================================================================
# MODELS METADATA ROUTER
# =============================================================================

models_router = APIRouter(prefix="/models", tags=["Model Registry"])


@models_router.get(
    "",
    response_model=ModelMetaResponse,
    summary="Model Registry",
    description="Returns metadata about all active ML models in the inference pipeline.",
)
async def get_model_metadata(api_key: str = Depends(get_api_key)):
    artifacts_exist = all(
        os.path.exists(p) for p in [
            "artifacts/model_radar.pkl",
            "artifacts/model_atrisk.pkl",
            "artifacts/model_cluster.pkl",
            "artifacts/model_anomaly.pkl",
        ]
    )

    return ModelMetaResponse(
        radar_model="MultiOutputRegressor(XGBRegressor) — Math/Reading/Writing scores",
        atrisk_model="RandomForestClassifier — Binary at-risk classification",
        cluster_model="KMeans(k=3) — Career trajectory clustering",
        anomaly_model="IsolationForest — Unusual profile detection",
        shap_explainer="SHAP TreeExplainer — Feature attribution for at-risk model",
        training_pipeline="sklearn Pipeline with OHE + StandardScaler preprocessing",
        framework="scikit-learn 1.3 / XGBoost 2.0",
        version="v2.4" if artifacts_exist else "NOT TRAINED — run data_ingestion.py",
    )
