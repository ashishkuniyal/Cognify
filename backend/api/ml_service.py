"""
ML Engine Service — Singleton wrapper around the 5-model inference pipeline.

This service layer decouples FastAPI routes from ML internals.
It loads all models once at startup (warm models = low latency)
and exposes clean methods for single & batch inference.
"""
from __future__ import annotations

import os
import sys
import time
import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils import load_object

logger = logging.getLogger("cognify.ml")


class MLService:
    """
    Singleton ML inference service. Initialized once at FastAPI startup.
    
    Encapsulates:
      - Model loading & health checking
      - Single-student inference with latency tracking
      - Batch inference with aggregated latency
      - What-If (prescriptive) simulation engine
    """

    _instance: Optional["MLService"] = None

    def __new__(cls) -> "MLService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self) -> None:
        """Load all models from disk. Called once at application startup."""
        if self._initialized:
            return

        try:
            logger.info("[INIT] Initializing ML Engine Service...")
            self._pipeline = PredictPipeline()

            # Verify artifacts exist
            required_artifacts = [
                "artifacts/preprocessor.pkl",
                "artifacts/model_radar.pkl",
                "artifacts/model_atrisk.pkl",
                "artifacts/model_cluster.pkl",
                "artifacts/model_anomaly.pkl"
            ]
            missing = [a for a in required_artifacts if not os.path.exists(a)]
            if missing:
                raise FileNotFoundError(
                    f"Missing model artifacts: {missing}. "
                    "Run: python -m src.components.data_ingestion"
                )

            self._models_loaded = True
            self._initialized = True
            logger.info("[OK] ML Engine Service initialized -- all models loaded.")

        except Exception as e:
            self._models_loaded = False
            logger.error(f"[ERROR] ML Engine initialization failed: {e}")
            raise

    @property
    def models_loaded(self) -> bool:
        return getattr(self, "_models_loaded", False)

    # ------------------------------------------------------------------
    # Core Inference
    # ------------------------------------------------------------------

    def predict_single(self, profile: dict) -> dict:
        """
        Run the full 5-engine inference pipeline on a single student profile.
        Returns a flat dict ready to be serialized into PredictionResponse.
        """
        start = time.perf_counter()

        data = CustomData(
            gender=profile["gender"],
            race_ethnicity=profile["race_ethnicity"],
            parental_level_of_education=profile["parental_level_of_education"],
            lunch=profile["lunch"],
            test_preparation_course=profile["test_preparation_course"],
            attendance_rate=profile.get("attendance_rate", 85.0),
            study_hours_per_week=profile.get("study_hours_per_week", 10.0),
            previous_gpa=profile.get("previous_gpa", 3.2),
            assignment_completion_rate=profile.get("assignment_completion_rate", 90.0)
        )
        pred_df = data.get_data_as_data_frame()
        results = self._pipeline.predict(pred_df)

        # Generate prescriptive advisor message
        results["advisor_message"] = self._run_prescriptive_engine(profile, pred_df, results)

        latency_ms = (time.perf_counter() - start) * 1000
        results["inference_latency_ms"] = round(latency_ms, 2)

        logger.info(
            f"Inference complete | math={results['math_score']:.1f} "
            f"at_risk={results['is_at_risk']} latency={latency_ms:.1f}ms"
        )
        return results

    def predict_batch(self, profiles: list[dict]) -> list[dict]:
        """
        Run inference on a batch of student profiles.
        Each profile is processed independently through the same pipeline.
        """
        logger.info(f"Batch inference started for {len(profiles)} profiles.")
        return [self.predict_single(p) for p in profiles]

    # ------------------------------------------------------------------
    # Prescriptive / What-If Engine
    # ------------------------------------------------------------------

    def _run_prescriptive_engine(
        self, original_profile: dict, pred_df: pd.DataFrame, base_results: dict
    ) -> str:
        """
        Simulates alternative scenarios to generate prescriptive advice based on weaknesses.
        """
        recs = []
        
        att = original_profile.get("attendance_rate", 100)
        study = original_profile.get("study_hours_per_week", 10)
        assign = original_profile.get("assignment_completion_rate", 100)
        prep = original_profile.get("test_preparation_course", "completed")
        
        if att < 80:
            recs.append("Improve attendance consistency (currently below 80%).")
        
        if study < 10:
            recs.append("Increase weekly study hours (currently below recommended 10h/week).")
            
        if assign < 80:
            recs.append("Ensure higher completion rate for assignments and homework.")
            
        if prep != "completed":
            try:
                sim_df = pred_df.copy()
                sim_df["test_preparation_course"] = "completed"
                sim_results = self._pipeline.predict(sim_df)
                boost = sim_results["math_score"] - base_results["math_score"]
                if boost > 0.5:
                    recs.append(f"Completing a test preparation course could boost Math by +{boost:.1f} pts.")
            except:
                pass
                
        if not recs:
            return "Student is demonstrating strong academic habits. Maintain current consistency."
            
        return "Recommendations: " + " | ".join(recs)


@lru_cache(maxsize=1)
def get_ml_service() -> MLService:
    """FastAPI dependency: returns the initialized MLService singleton."""
    return MLService()
