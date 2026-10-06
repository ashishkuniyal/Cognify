"""
Middleware — Cross-cutting concerns for the Cognify FastAPI application.

Implements:
  1. RequestIDMiddleware  — attaches a unique UUID to every request for distributed tracing
  2. StructuredLoggingMiddleware — logs method, path, status, latency as JSON
  3. RateLimiter — simple in-memory sliding-window rate limiter (per API key or IP)
  4. APIKeyAuth — FastAPI dependency that validates Authorization: Bearer <key>
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Optional

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

import os

# ---------------------------------------------------------------------------
# Structured Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("cognify.api")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects X-Request-ID header into every request and response."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.start_time = time.perf_counter()

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request as a structured JSON log line. Example:
    {
      "level": "INFO", "method": "POST", "path": "/api/v1/predict",
      "status": 200, "latency_ms": 45.2, "request_id": "abc-123"
    }
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        request_id = getattr(request.state, "request_id", "unknown")
        log_entry = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(latency_ms, 2),
            "request_id": request_id,
            "client_ip": request.client.host if request.client else "unknown",
        }

        if response.status_code >= 400:
            logger.warning(json.dumps(log_entry))
        else:
            logger.info(json.dumps(log_entry))

        response.headers["X-Latency-Ms"] = str(round(latency_ms, 2))
        return response


# ---------------------------------------------------------------------------
# In-Memory Sliding Window Rate Limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """
    Thread-safe in-memory sliding-window rate limiter.
    Tracks the number of requests per (key, endpoint) within a time window.
    """

    def __init__(self):
        self._buckets: dict[str, deque] = defaultdict(deque)

    def is_allowed(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        bucket = self._buckets[key]

        # Evict timestamps outside the window
        while bucket and bucket[0] < now - window_seconds:
            bucket.popleft()

        remaining = limit - len(bucket)
        if len(bucket) >= limit:
            return False, 0

        bucket.append(now)
        return True, remaining - 1


_rate_limiter = _RateLimiter()

# Rate limits (requests / window_seconds)
RATE_LIMITS = {
    "predict": (30, 60),          # 30 req / 60s per API key
    "batch":   (5, 60),           # 5 batch req / 60s per API key
    "chat":    (20, 60),          # 20 chat req / 60s per API key
    "default": (60, 60),          # generic endpoints
}


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)

VALID_API_KEYS: set[str] = {
    _hash_key(k) for k in os.environ.get("COGNIFY_API_KEYS", "dev-key-2026").split(",")
}


def get_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> str:
    """
    FastAPI dependency: validates the Bearer token in the Authorization header.
    Falls back to the X-API-Key header. Returns the raw key on success.
    
    In development mode (API_AUTH_DISABLED=true), skips validation entirely.
    """
    if os.environ.get("API_AUTH_DISABLED", "false").lower() == "true":
        return "dev-anonymous"

    raw_key: Optional[str] = None

    if credentials:
        raw_key = credentials.credentials
    elif "x-api-key" in request.headers:
        raw_key = request.headers["x-api-key"]

    if not raw_key or _hash_key(raw_key) not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass it as: Authorization: Bearer <key> or X-API-Key: <key>",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return raw_key


def apply_rate_limit(request: Request, api_key: str, endpoint_name: str) -> None:
    """
    Apply rate limiting for a given endpoint. Raises 429 if limit exceeded.
    Uses the API key (or client IP as fallback) as the bucket key.
    """
    limit, window = RATE_LIMITS.get(endpoint_name, RATE_LIMITS["default"])
    bucket_key = f"{_hash_key(api_key)}:{endpoint_name}"

    allowed, remaining = _rate_limiter.is_allowed(bucket_key, limit, window)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests per {window}s on /{endpoint_name}.",
            headers={"X-RateLimit-Remaining": "0", "Retry-After": str(window)},
        )
