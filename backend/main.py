"""
Cognify FastAPI Application — Main Entrypoint

Production-grade FastAPI application exposing the Cognify ML inference 
platform as a fully-documented REST API.

Run with:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Swagger UI auto-generated at: http://localhost:8000/docs
ReDoc at:                      http://localhost:8000/redoc
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Logging — structured JSON to stdout
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "msg": %(message)s}',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("cognify.main")

# ---------------------------------------------------------------------------
# Application Lifespan (startup & shutdown hooks)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    - On startup: initialize DB tables and warm up ML models
    - On shutdown: clean up resources
    """
    logger.info('"Cognify API starting up..."')

    # Init database
    from api.database import init_db
    init_db()
    logger.info('"Database tables initialized."')

    # Warm up ML models (load into memory once)
    from api.ml_service import get_ml_service
    ml = get_ml_service()
    try:
        ml.initialize()
    except FileNotFoundError as e:
        logger.warning(f'"ML models not found -- {e}. Train first."')
    except Exception as e:
        logger.error(f'"ML initialization error: {e}"')

    logger.info('"[READY] Cognify API is ready."')
    yield

    logger.info('"Cognify API shutting down."')


# ---------------------------------------------------------------------------
# FastAPI App Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cognify — Intelligent EdTech API",
    description="""
## Cognify ML-Powered Student Analytics Platform

A production-grade REST API for student performance prediction, career trajectory mapping,
and at-risk early intervention — built with FastAPI + scikit-learn + XGBoost.

### 🔑 Authentication
All endpoints require a Bearer token or `X-API-Key` header.  
**Development key**: `dev-key-2026`

### 🤖 ML Engine Architecture
| Engine | Algorithm | Task |
|--------|-----------|------|
| Radar  | XGBoost MultiOutput Regressor | Predict Math / Reading / Writing scores |
| At-Risk | Random Forest Classifier | Identify students at risk of failure |
| Career | K-Means (k=3) | Cluster students into career paths |
| Anomaly | Isolation Forest | Detect unusual student profiles |
| XAI | SHAP TreeExplainer | Explain primary risk factors |

### 📊 API Versioning
All production endpoints are versioned under `/api/v1/`.

### 🔒 Rate Limits
- `/predict`: 30 req/min
- `/predict/batch`: 5 req/min  
- `/chat`: 20 req/min
""",
    version="2.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={
        "name": "Cognify Engineering",
        "url": "https://github.com/ashishkuniyal/Intelligent-EdTech-Platform",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# ---------------------------------------------------------------------------
# CORS — Allow frontend origins
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:4173,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Latency-Ms", "X-RateLimit-Remaining"],
)

# ---------------------------------------------------------------------------
# Custom Middleware (must add AFTER CORSMiddleware)
# ---------------------------------------------------------------------------

from api.middleware import RequestIDMiddleware, StructuredLoggingMiddleware

app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

# ---------------------------------------------------------------------------
# Register Routers at /api/v1/
# ---------------------------------------------------------------------------

from api.routes import analytics_router, chat_router, models_router, prediction_router

API_PREFIX = "/api/v1"

app.include_router(prediction_router, prefix=API_PREFIX)
app.include_router(analytics_router, prefix=API_PREFIX)
app.include_router(chat_router, prefix=API_PREFIX)
app.include_router(models_router, prefix=API_PREFIX)

# ---------------------------------------------------------------------------
# Root & Health Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"], summary="Root — API Info")
async def root():
    return {
        "name": "Cognify Intelligent EdTech API",
        "version": "2.4.0",
        "docs": "/docs",
        "health": "/health",
        "api_prefix": "/api/v1",
        "endpoints": [
            "POST /api/v1/predict",
            "POST /api/v1/predict/batch",
            "GET  /api/v1/analytics",
            "POST /api/v1/chat",
            "GET  /api/v1/models",
        ]
    }


@app.get("/health", tags=["Health"], summary="Health Check")
async def health_check():
    from api.ml_service import get_ml_service
    from api.database import SessionLocal
    from api.schemas import HealthResponse

    ml = get_ml_service()

    # Quick DB check
    db_ok = False
    try:
        db = SessionLocal()
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
        db.close()
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if ml.models_loaded and db_ok else "degraded",
        models_loaded=ml.models_loaded,
        database_ok=db_ok,
        version="2.4.0",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# ---------------------------------------------------------------------------
# Global Exception Handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f'"Unhandled exception | request_id={request_id} | error={str(exc)}"')
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "request_id": request_id,
        },
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
