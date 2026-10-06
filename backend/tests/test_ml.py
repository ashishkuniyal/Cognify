import os
import pytest
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

def test_custom_data_to_dataframe():
    data = CustomData(
        gender="female",
        race_ethnicity="group C",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        attendance_rate=85.0,
        study_hours_per_week=12.0,
        previous_gpa=3.2,
        assignment_completion_rate=90.0
    )
    df = data.get_data_as_data_frame()
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 9)
    assert df.iloc[0]['attendance_rate'] == 85.0

def test_predict_pipeline():
    pipeline = PredictPipeline()
    
    # We can only test this if models exist
    if os.path.exists("artifacts/preprocessor.pkl"):
        data = CustomData(
            gender="female",
            race_ethnicity="group C",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            attendance_rate=85.0,
            study_hours_per_week=12.0,
            previous_gpa=3.2,
            assignment_completion_rate=90.0
        )
        df = data.get_data_as_data_frame()
        
        results = pipeline.predict(df)
        
        assert "math_score" in results
        assert "overall_score" in results
        assert "risk_level" in results
        assert "top_risk_factor" in results
